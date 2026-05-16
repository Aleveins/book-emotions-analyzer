package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

type LLMStatus struct {
	Available bool   `json:"available"`
	Provider  string `json:"provider"`
	Model     string `json:"model"`
	Reason    string `json:"reason,omitempty"`
}

type LLMStatusService struct {
	provider  string
	model     string
	ollamaURL string
	client    *http.Client
}

func NewLLMStatusService(provider, model, ollamaURL string, timeout time.Duration) *LLMStatusService {
	return &LLMStatusService{
		provider:  strings.ToLower(strings.TrimSpace(provider)),
		model:     strings.TrimSpace(model),
		ollamaURL: strings.TrimRight(strings.TrimSpace(ollamaURL), "/"),
		client: &http.Client{
			Timeout: timeout,
		},
	}
}

func (s *LLMStatusService) Status(ctx context.Context) LLMStatus {
	status := LLMStatus{
		Provider: s.provider,
		Model:    s.model,
	}

	switch s.provider {
	case "ollama":
		return s.ollamaStatus(ctx, status)
	case "openai":
		status.Available = os.Getenv("OPENAI_API_KEY") != ""
		if !status.Available {
			status.Reason = "OPENAI_API_KEY is not configured"
		}
		return status
	default:
		status.Reason = fmt.Sprintf("unsupported LLM provider: %s", s.provider)
		return status
	}
}

func (s *LLMStatusService) ollamaStatus(ctx context.Context, status LLMStatus) LLMStatus {
	if s.ollamaURL == "" {
		status.Reason = "Ollama URL is not configured"
		return status
	}
	if s.model == "" {
		status.Reason = "LLM model is not configured"
		return status
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.ollamaURL+"/api/tags", nil)
	if err != nil {
		status.Reason = err.Error()
		return status
	}

	resp, err := s.client.Do(req)
	if err != nil {
		status.Reason = err.Error()
		return status
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		status.Reason = fmt.Sprintf("Ollama returned HTTP %d", resp.StatusCode)
		return status
	}

	var payload struct {
		Models []struct {
			Name  string `json:"name"`
			Model string `json:"model"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		status.Reason = "failed to decode Ollama model list: " + err.Error()
		return status
	}

	for _, model := range payload.Models {
		if model.Name == s.model || model.Model == s.model {
			return s.ollamaChatStatus(ctx, status)
		}
	}

	status.Reason = fmt.Sprintf("model %s is not pulled in Ollama", s.model)
	return status
}

func (s *LLMStatusService) ollamaChatStatus(ctx context.Context, status LLMStatus) LLMStatus {
	payload := map[string]any{
		"model":  s.model,
		"stream": false,
		"format": "json",
		"options": map[string]any{
			"temperature": 0,
			"num_predict": 8,
		},
		"messages": []map[string]string{
			{"role": "system", "content": "Return valid JSON only."},
			{"role": "user", "content": `Return exactly {"ok":true}.`},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		status.Reason = err.Error()
		return status
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.ollamaURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		status.Reason = err.Error()
		return status
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		status.Reason = "Ollama chat check failed: " + err.Error()
		return status
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		preview, _ := io.ReadAll(io.LimitReader(resp.Body, 300))
		status.Reason = fmt.Sprintf("Ollama chat check returned HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(preview)))
		return status
	}

	var response struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
		Error string `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		status.Reason = "failed to decode Ollama chat check: " + err.Error()
		return status
	}
	if response.Error != "" {
		status.Reason = "Ollama chat check failed: " + response.Error
		return status
	}

	var parsed struct {
		OK bool `json:"ok"`
	}
	if err := json.Unmarshal([]byte(response.Message.Content), &parsed); err != nil || !parsed.OK {
		status.Reason = "Ollama chat check returned an invalid response"
		return status
	}

	status.Available = true
	return status
}
