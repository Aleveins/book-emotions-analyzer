package grpcserver

import (
	"context"
	"errors"
	"log/slog"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	pb "github.com/Aleveins/book-emotions-analyzer/pkg/pb/text_storage"
	"github.com/Aleveins/book-emotions-analyzer/services/text-storage/internal/model"
	"github.com/Aleveins/book-emotions-analyzer/services/text-storage/internal/repository"
)

type Server struct {
	pb.UnimplementedTextStorageServiceServer
	textRepo   *repository.TextRepository
	resultRepo *repository.ResultRepository
}

func NewServer(textRepo *repository.TextRepository, resultRepo *repository.ResultRepository) *Server {
	return &Server{
		textRepo:   textRepo,
		resultRepo: resultRepo,
	}
}

func (s *Server) SaveText(ctx context.Context, req *pb.SaveTextRequest) (*pb.SaveTextResponse, error) {
	if req.GetUserId() == "" {
		return nil, status.Error(codes.InvalidArgument, "user_id is required")
	}
	if req.GetTitle() == "" {
		return nil, status.Error(codes.InvalidArgument, "title is required")
	}
	if len(req.GetContent()) == 0 {
		return nil, status.Error(codes.InvalidArgument, "content is required")
	}
	format := req.GetFormat()
	if format == "" {
		format = "txt"
	}

	text := &model.Text{
		UserID:   req.GetUserId(),
		Title:    req.GetTitle(),
		Filename: req.GetFilename(),
		Content:  req.GetContent(),
		Format:   format,
	}

	if err := s.textRepo.Create(ctx, text); err != nil {
		slog.Error("failed to save text", "error", err)
		return nil, status.Error(codes.Internal, "failed to save text")
	}

	slog.Info("text saved", "text_id", text.ID, "user_id", text.UserID)

	return &pb.SaveTextResponse{
		TextId: text.ID,
	}, nil
}

func (s *Server) GetText(ctx context.Context, req *pb.GetTextRequest) (*pb.GetTextResponse, error) {
	if req.GetTextId() == "" {
		return nil, status.Error(codes.InvalidArgument, "text_id is required")
	}

	text, err := s.textRepo.GetByID(ctx, req.GetTextId())
	if err != nil {
		if errors.Is(err, repository.ErrTextNotFound) {
			return nil, status.Error(codes.NotFound, "text not found")
		}
		slog.Error("failed to get text", "error", err, "text_id", req.GetTextId())
		return nil, status.Error(codes.Internal, "failed to get text")
	}

	return &pb.GetTextResponse{
		Text: textModelToProto(text),
	}, nil
}

func (s *Server) ListTexts(ctx context.Context, req *pb.ListTextsRequest) (*pb.ListTextsResponse, error) {
	if req.GetUserId() == "" {
		return nil, status.Error(codes.InvalidArgument, "user_id is required")
	}

	limit := int(req.GetLimit())
	if limit <= 0 {
		limit = 20
	}
	offset := int(req.GetOffset())
	if offset < 0 {
		offset = 0
	}

	texts, total, err := s.textRepo.ListByUserID(ctx, req.GetUserId(), limit, offset)
	if err != nil {
		slog.Error("failed to list texts", "error", err, "user_id", req.GetUserId())
		return nil, status.Error(codes.Internal, "failed to list texts")
	}

	pbTexts := make([]*pb.Text, 0, len(texts))
	for _, t := range texts {
		pbTexts = append(pbTexts, textModelToProto(t))
	}

	return &pb.ListTextsResponse{
		Texts: pbTexts,
		Total: int32(total),
	}, nil
}

func (s *Server) DeleteText(ctx context.Context, req *pb.DeleteTextRequest) (*pb.DeleteTextResponse, error) {
	if req.GetTextId() == "" {
		return nil, status.Error(codes.InvalidArgument, "text_id is required")
	}

	if err := s.textRepo.Delete(ctx, req.GetTextId()); err != nil {
		if errors.Is(err, repository.ErrTextNotFound) {
			return nil, status.Error(codes.NotFound, "text not found")
		}
		slog.Error("failed to delete text", "error", err, "text_id", req.GetTextId())
		return nil, status.Error(codes.Internal, "failed to delete text")
	}

	slog.Info("text deleted", "text_id", req.GetTextId())

	return &pb.DeleteTextResponse{}, nil
}

func (s *Server) SaveAnalysisResult(ctx context.Context, req *pb.SaveAnalysisResultRequest) (*pb.SaveAnalysisResultResponse, error) {
	if req.GetTextId() == "" {
		return nil, status.Error(codes.InvalidArgument, "text_id is required")
	}
	if req.GetJobId() == "" {
		return nil, status.Error(codes.InvalidArgument, "job_id is required")
	}
	if len(req.GetResultJson()) == 0 {
		return nil, status.Error(codes.InvalidArgument, "result_json is required")
	}

	result := &model.AnalysisResult{
		TextID:     req.GetTextId(),
		JobID:      req.GetJobId(),
		ResultJSON: req.GetResultJson(),
	}

	if err := s.resultRepo.Save(ctx, result); err != nil {
		slog.Error("failed to save analysis result", "error", err, "job_id", req.GetJobId())
		return nil, status.Error(codes.Internal, "failed to save analysis result")
	}

	slog.Info("analysis result saved", "result_id", result.ID, "job_id", result.JobID, "size", len(result.ResultJSON))

	return &pb.SaveAnalysisResultResponse{
		ResultId: result.ID,
	}, nil
}

func (s *Server) GetAnalysisResult(ctx context.Context, req *pb.GetAnalysisResultRequest) (*pb.GetAnalysisResultResponse, error) {
	if req.GetJobId() == "" {
		return nil, status.Error(codes.InvalidArgument, "job_id is required")
	}

	result, err := s.resultRepo.GetByJobID(ctx, req.GetJobId())
	if err != nil {
		if errors.Is(err, repository.ErrResultNotFound) {
			return nil, status.Error(codes.NotFound, "analysis result not found")
		}
		slog.Error("failed to get analysis result", "error", err, "job_id", req.GetJobId())
		return nil, status.Error(codes.Internal, "failed to get analysis result")
	}

	return &pb.GetAnalysisResultResponse{
		Id:         result.ID,
		TextId:     result.TextID,
		JobId:      result.JobID,
		ResultJson: result.ResultJSON,
		CreatedAt:  result.CreatedAt.Format("2006-01-02T15:04:05Z07:00"),
	}, nil
}

func textModelToProto(t *model.Text) *pb.Text {
	return &pb.Text{
		Id:        t.ID,
		UserId:    t.UserID,
		Title:     t.Title,
		Filename:  t.Filename,
		Content:   t.Content,
		Format:    t.Format,
		CreatedAt: t.CreatedAt.Format("2006-01-02T15:04:05Z07:00"),
	}
}
