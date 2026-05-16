package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/redis/go-redis/v9"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/Aleveins/book-emotions-analyzer/pkg/pb/text_storage"
	"github.com/Aleveins/book-emotions-analyzer/services/job-processor/internal/config"
	"github.com/Aleveins/book-emotions-analyzer/services/job-processor/internal/handler"
	"github.com/Aleveins/book-emotions-analyzer/services/job-processor/internal/service"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)

	cfg, err := config.Load()
	if err != nil {
		slog.Error("failed to load config", "error", err)
		os.Exit(1)
	}

	rdb := redis.NewClient(&redis.Options{
		Addr: cfg.RedisAddr,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := rdb.Ping(ctx).Err(); err != nil {
		slog.Error("failed to connect to redis", "error", err)
		os.Exit(1)
	}
	slog.Info("connected to redis", "addr", cfg.RedisAddr)

	nc, err := nats.Connect(cfg.NatsURL)
	if err != nil {
		slog.Error("failed to connect to nats", "error", err)
		os.Exit(1)
	}
	defer nc.Close()
	slog.Info("connected to nats", "url", cfg.NatsURL)

	const maxGRPCMessageSize = 64 * 1024 * 1024 // 64 МБ для больших PDF
	grpcConn, err := grpc.NewClient(
		cfg.TextStorageAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(maxGRPCMessageSize),
			grpc.MaxCallSendMsgSize(maxGRPCMessageSize),
		),
	)
	if err != nil {
		slog.Error("failed to connect to text-storage grpc", "error", err)
		os.Exit(1)
	}
	defer grpcConn.Close()
	slog.Info("connected to text-storage grpc", "addr", cfg.TextStorageAddr)

	textStorageClient := pb.NewTextStorageServiceClient(grpcConn)

	jobService := service.NewJobService(rdb, nc, textStorageClient)
	llmStatusService := service.NewLLMStatusService(
		cfg.LLMProvider,
		cfg.LLMModel,
		cfg.OllamaURL,
		time.Duration(cfg.LLMStatusTimeoutSeconds)*time.Second,
	)
	router := handler.NewRouter(jobService, llmStatusService, cfg.JWTSecret)

	srv := &http.Server{
		Addr:         ":" + cfg.ServerPort,
		Handler:      router,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		slog.Info("starting server", "port", cfg.ServerPort)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Error("server failed", "error", err)
			os.Exit(1)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	sig := <-quit
	slog.Info("shutting down server", "signal", sig.String())

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		slog.Error("server forced to shutdown", "error", err)
		os.Exit(1)
	}

	slog.Info("server stopped gracefully")
}
