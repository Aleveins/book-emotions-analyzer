package repository

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/Aleveins/book-emotions-analyzer/services/text-storage/internal/model"
)

var (
	ErrResultNotFound = errors.New("analysis result not found")
)

type ResultRepository struct {
	pool *pgxpool.Pool
}

func NewResultRepository(pool *pgxpool.Pool) *ResultRepository {
	return &ResultRepository{pool: pool}
}

func (r *ResultRepository) Save(ctx context.Context, result *model.AnalysisResult) error {
	query := `
		INSERT INTO analysis_results (text_id, job_id, result_json)
		VALUES ($1, $2, $3)
		RETURNING id, created_at`

	err := r.pool.QueryRow(ctx, query, result.TextID, result.JobID, result.ResultJSON).
		Scan(&result.ID, &result.CreatedAt)
	if err != nil {
		return fmt.Errorf("insert analysis result: %w", err)
	}

	return nil
}

func (r *ResultRepository) GetByJobID(ctx context.Context, jobID string) (*model.AnalysisResult, error) {
	query := `
		SELECT id, text_id, job_id, result_json, created_at
		FROM analysis_results
		WHERE job_id = $1`

	result := &model.AnalysisResult{}
	err := r.pool.QueryRow(ctx, query, jobID).
		Scan(&result.ID, &result.TextID, &result.JobID, &result.ResultJSON, &result.CreatedAt)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrResultNotFound
		}
		return nil, fmt.Errorf("get analysis result by job_id: %w", err)
	}

	return result, nil
}
