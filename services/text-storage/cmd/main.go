package main

import (
	"context"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"google.golang.org/grpc"

	pb "github.com/Aleveins/book-emotions-analyzer/pkg/pb/text_storage"
	"github.com/Aleveins/book-emotions-analyzer/services/text-storage/internal/config"
	grpcserver "github.com/Aleveins/book-emotions-analyzer/services/text-storage/internal/grpcserver"
	"github.com/Aleveins/book-emotions-analyzer/services/text-storage/internal/repository"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)

	cfg, err := config.Load()
	if err != nil {
		slog.Error("failed to load config", "error", err)
		os.Exit(1)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	pool, err := pgxpool.New(ctx, cfg.DatabaseURL())
	if err != nil {
		slog.Error("failed to connect to database", "error", err)
		os.Exit(1)
	}
	defer pool.Close()

	if err := pool.Ping(ctx); err != nil {
		slog.Error("failed to ping database", "error", err)
		os.Exit(1)
	}
	slog.Info("connected to database")

	textRepo := repository.NewTextRepository(pool)
	resultRepo := repository.NewResultRepository(pool)
	srv := grpcserver.NewServer(textRepo, resultRepo)

	const maxGRPCMessageSize = 64 * 1024 * 1024 // 64 МБ для больших PDF
	grpcSrv := grpc.NewServer(
		grpc.MaxRecvMsgSize(maxGRPCMessageSize),
		grpc.MaxSendMsgSize(maxGRPCMessageSize),
	)
	pb.RegisterTextStorageServiceServer(grpcSrv, srv)

	lis, err := net.Listen("tcp", ":"+cfg.GRPCPort)
	if err != nil {
		slog.Error("failed to listen", "error", err, "port", cfg.GRPCPort)
		os.Exit(1)
	}

	go func() {
		slog.Info("starting gRPC server", "port", cfg.GRPCPort)
		if err := grpcSrv.Serve(lis); err != nil {
			slog.Error("gRPC server failed", "error", err)
			os.Exit(1)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	sig := <-quit
	slog.Info("shutting down server", "signal", sig.String())

	grpcSrv.GracefulStop()

	slog.Info("server stopped gracefully")
}
