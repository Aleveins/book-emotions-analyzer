package service

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
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

	if !status.Available {
		t.Fatalf("expected model to be available, got reason: %s", status.Reason)
	}
}

func TestLLMStatusUnavailableWhenOllamaModelIsMissing(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"models":[{"name":"llama3.2:3b","model":"llama3.2:3b"}]}`))
	}))
	defer server.Close()

	service := NewLLMStatusService("ollama", "qwen2.5:7b", server.URL, time.Second)
	status := service.Status(context.Background())

	if status.Available {
		t.Fatal("expected model to be unavailable")
	}
	if status.Reason == "" {
		t.Fatal("expected unavailable status to include a reason")
	}
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

	if status.Available {
		t.Fatal("expected model to be unavailable")
	}
	if status.Reason == "" {
		t.Fatal("expected unavailable status to include a reason")
	}
}
