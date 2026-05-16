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
	ErrTextNotFound = errors.New("text not found")
)

type TextRepository struct {
	pool *pgxpool.Pool
}

func NewTextRepository(pool *pgxpool.Pool) *TextRepository {
	return &TextRepository{pool: pool}
}

func (r *TextRepository) Create(ctx context.Context, text *model.Text) error {
	query := `
		INSERT INTO texts (user_id, title, filename, content, format)
		VALUES ($1, $2, $3, $4, $5)
		RETURNING id, created_at`

	err := r.pool.QueryRow(ctx, query,
		text.UserID, text.Title, text.Filename, text.Content, text.Format,
	).Scan(&text.ID, &text.CreatedAt)
	if err != nil {
		return fmt.Errorf("create text: %w", err)
	}

	return nil
}

func (r *TextRepository) GetByID(ctx context.Context, id string) (*model.Text, error) {
	query := `
		SELECT id, user_id, title, filename, content, format, created_at
		FROM texts
		WHERE id = $1`

	text := &model.Text{}
	err := r.pool.QueryRow(ctx, query, id).
		Scan(&text.ID, &text.UserID, &text.Title, &text.Filename, &text.Content, &text.Format, &text.CreatedAt)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrTextNotFound
		}
		return nil, fmt.Errorf("get text by id: %w", err)
	}

	return text, nil
}

func (r *TextRepository) ListByUserID(ctx context.Context, userID string, limit, offset int) ([]*model.Text, int, error) {
	countQuery := `SELECT COUNT(*) FROM texts WHERE user_id = $1`
	var total int
	if err := r.pool.QueryRow(ctx, countQuery, userID).Scan(&total); err != nil {
		return nil, 0, fmt.Errorf("count texts: %w", err)
	}

	query := `
		SELECT id, user_id, title, filename, content, format, created_at
		FROM texts
		WHERE user_id = $1
		ORDER BY created_at DESC
		LIMIT $2 OFFSET $3`

	rows, err := r.pool.Query(ctx, query, userID, limit, offset)
	if err != nil {
		return nil, 0, fmt.Errorf("list texts: %w", err)
	}
	defer rows.Close()

	var texts []*model.Text
	for rows.Next() {
		t := &model.Text{}
		if err := rows.Scan(&t.ID, &t.UserID, &t.Title, &t.Filename, &t.Content, &t.Format, &t.CreatedAt); err != nil {
			return nil, 0, fmt.Errorf("scan text row: %w", err)
		}
		texts = append(texts, t)
	}

	if err := rows.Err(); err != nil {
		return nil, 0, fmt.Errorf("iterate text rows: %w", err)
	}

	return texts, total, nil
}

func (r *TextRepository) Delete(ctx context.Context, id string) error {
	query := `DELETE FROM texts WHERE id = $1`

	ct, err := r.pool.Exec(ctx, query, id)
	if err != nil {
		return fmt.Errorf("delete text: %w", err)
	}

	if ct.RowsAffected() == 0 {
		return ErrTextNotFound
	}

	return nil
}
