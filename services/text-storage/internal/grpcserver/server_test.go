package grpcserver

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/Aleveins/book-emotions-analyzer/services/text-storage/internal/model"
)

func TestTextModelToProto(t *testing.T) {
	createdAt := time.Date(2026, 5, 18, 12, 30, 0, 0, time.UTC)
	text := &model.Text{
		ID:        "text-1",
		UserID:    "user-1",
		Title:     "Book",
		Filename:  "book.txt",
		Content:   []byte("content"),
		Format:    "txt",
		CreatedAt: createdAt,
	}

	got := textModelToProto(text)

	assert.Equal(t, text.ID, got.GetId())
	assert.Equal(t, text.UserID, got.GetUserId())
	assert.Equal(t, text.Title, got.GetTitle())
	assert.Equal(t, text.Filename, got.GetFilename())
	assert.Equal(t, text.Content, got.GetContent())
	assert.Equal(t, text.Format, got.GetFormat())
	assert.Equal(t, "2026-05-18T12:30:00Z", got.GetCreatedAt())
}
