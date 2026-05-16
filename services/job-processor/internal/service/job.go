package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/nats-io/nats.go"
	"github.com/redis/go-redis/v9"

	pb "github.com/Aleveins/book-emotions-analyzer/pkg/pb/text_storage"
)

var (
	ErrJobNotFound    = errors.New("job not found")
	ErrJobNotComplete = errors.New("job is not completed")
)

type Job struct {
	ID                   string `json:"id"`
	TextID               string `json:"text_id"`
	UserID               string `json:"user_id"`
	Title                string `json:"title"`
	Status               string `json:"status"`
	CreatedAt            string `json:"created_at"`
	UpdatedAt            string `json:"updated_at"`
	Error                string `json:"error"`
	UseLLM               bool   `json:"use_llm"`
	UseParagraphAnalysis bool   `json:"use_paragraph_analysis"`
}

type AnalysisResultResponse struct {
	ID        string          `json:"id"`
	TextID    string          `json:"text_id"`
	JobID     string          `json:"job_id"`
	Result    json.RawMessage `json:"result"`
	CreatedAt string          `json:"created_at"`
}

type JobService struct {
	rdb         *redis.Client
	nc          *nats.Conn
	textStorage pb.TextStorageServiceClient
}

func NewJobService(rdb *redis.Client, nc *nats.Conn, textStorage pb.TextStorageServiceClient) *JobService {
	return &JobService{
		rdb:         rdb,
		nc:          nc,
		textStorage: textStorage,
	}
}

func (s *JobService) CreateJob(ctx context.Context, userID, title, filename, format string, content []byte, useLLM bool, useParagraphAnalysis bool) (*Job, error) {
	saveResp, err := s.textStorage.SaveText(ctx, &pb.SaveTextRequest{
		UserId:   userID,
		Title:    title,
		Filename: filename,
		Content:  content,
		Format:   format,
	})
	if err != nil {
		return nil, fmt.Errorf("save text via grpc: %w", err)
	}

	now := time.Now().UTC().Format(time.RFC3339)
	job := &Job{
		ID:                   uuid.New().String(),
		TextID:               saveResp.GetTextId(),
		UserID:               userID,
		Title:                title,
		Status:               "pending",
		CreatedAt:            now,
		UpdatedAt:            now,
		Error:                "",
		UseLLM:               useLLM,
		UseParagraphAnalysis: useParagraphAnalysis,
	}

	jobJSON, err := json.Marshal(job)
	if err != nil {
		return nil, fmt.Errorf("marshal job: %w", err)
	}

	jobKey := fmt.Sprintf("job:%s", job.ID)
	if err := s.rdb.Set(ctx, jobKey, jobJSON, 0).Err(); err != nil {
		return nil, fmt.Errorf("save job to redis: %w", err)
	}

	userJobsKey := fmt.Sprintf("user_jobs:%s", userID)
	if err := s.rdb.SAdd(ctx, userJobsKey, job.ID).Err(); err != nil {
		return nil, fmt.Errorf("add job to user set: %w", err)
	}

	if err := s.nc.Publish("jobs.process", []byte(job.ID)); err != nil {
		return nil, fmt.Errorf("publish job to nats: %w", err)
	}

	return job, nil
}

func (s *JobService) GetJob(ctx context.Context, jobID, userID string) (*Job, error) {
	jobKey := fmt.Sprintf("job:%s", jobID)
	jobJSON, err := s.rdb.Get(ctx, jobKey).Result()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return nil, ErrJobNotFound
		}
		return nil, fmt.Errorf("get job from redis: %w", err)
	}

	var job Job
	if err := json.Unmarshal([]byte(jobJSON), &job); err != nil {
		return nil, fmt.Errorf("unmarshal job: %w", err)
	}

	if job.UserID != userID {
		return nil, ErrJobNotFound
	}

	return &job, nil
}

func (s *JobService) ListJobs(ctx context.Context, userID string) ([]*Job, error) {
	userJobsKey := fmt.Sprintf("user_jobs:%s", userID)
	jobIDs, err := s.rdb.SMembers(ctx, userJobsKey).Result()
	if err != nil {
		return nil, fmt.Errorf("get user job ids: %w", err)
	}

	jobs := make([]*Job, 0, len(jobIDs))
	for _, jobID := range jobIDs {
		jobKey := fmt.Sprintf("job:%s", jobID)
		jobJSON, err := s.rdb.Get(ctx, jobKey).Result()
		if err != nil {
			if errors.Is(err, redis.Nil) {
				continue
			}
			return nil, fmt.Errorf("get job from redis: %w", err)
		}

		var job Job
		if err := json.Unmarshal([]byte(jobJSON), &job); err != nil {
			return nil, fmt.Errorf("unmarshal job: %w", err)
		}
		jobs = append(jobs, &job)
	}

	return jobs, nil
}

func (s *JobService) GetResult(ctx context.Context, jobID, userID string) (*AnalysisResultResponse, error) {
	job, err := s.GetJob(ctx, jobID, userID)
	if err != nil {
		return nil, err
	}

	if job.Status != "completed" {
		return nil, ErrJobNotComplete
	}

	resp, err := s.textStorage.GetAnalysisResult(ctx, &pb.GetAnalysisResultRequest{
		JobId: jobID,
	})
	if err != nil {
		return nil, fmt.Errorf("get analysis result via grpc: %w", err)
	}

	return &AnalysisResultResponse{
		ID:        resp.GetId(),
		TextID:    resp.GetTextId(),
		JobID:     resp.GetJobId(),
		Result:    resp.GetResultJson(),
		CreatedAt: resp.GetCreatedAt(),
	}, nil
}
