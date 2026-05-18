package service

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestLLMStatusAvailableWhenOllamaModelExists(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"models":[{"name":"qwen2.5:7b","model":"qwen2.5:7b"}]}`))
		case "/api/chat":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"message":{"content":"{\"ok\":true}"}}`))
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer server.Close()

	service := NewLLMStatusService("ollama", "qwen2.5:7b", server.URL, time.Second)
	status := service.Status(context.Background())

	assert.True(t, status.Available, "reason: %s", status.Reason)
}

func TestLLMStatusUnavailableWhenOllamaModelIsMissing(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"models":[{"name":"llama3.2:3b","model":"llama3.2:3b"}]}`))
	}))
	defer server.Close()

	service := NewLLMStatusService("ollama", "qwen2.5:7b", server.URL, time.Second)
	status := service.Status(context.Background())

	assert.False(t, status.Available)
	assert.NotEmpty(t, status.Reason)
}

func TestLLMStatusUnavailableWhenOllamaChatCheckFails(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"models":[{"name":"qwen2.5:7b","model":"qwen2.5:7b"}]}`))
		case "/api/chat":
			http.Error(w, "model failed", http.StatusInternalServerError)
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer server.Close()

	service := NewLLMStatusService("ollama", "qwen2.5:7b", server.URL, time.Second)
	status := service.Status(context.Background())

	assert.False(t, status.Available)
	assert.NotEmpty(t, status.Reason)
}
