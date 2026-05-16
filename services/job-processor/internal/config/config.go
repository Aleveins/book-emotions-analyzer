package config

import (
	"fmt"
	"os"
)

type Config struct {
	ServerPort              string
	RedisAddr               string
	NatsURL                 string
	TextStorageAddr         string
	JWTSecret               string
	LLMProvider             string
	LLMModel                string
	OllamaURL               string
	LLMStatusTimeoutSeconds int
}

func Load() (*Config, error) {
	cfg := &Config{
		ServerPort:              getEnv("SERVER_PORT", "8082"),
		RedisAddr:               getEnv("REDIS_ADDR", "localhost:6379"),
		NatsURL:                 getEnv("NATS_URL", "nats://localhost:4222"),
		TextStorageAddr:         getEnv("TEXT_STORAGE_ADDR", "localhost:50051"),
		JWTSecret:               getEnv("JWT_SECRET", ""),
		LLMProvider:             getEnv("LLM_PROVIDER", "ollama"),
		LLMModel:                getEnv("LLM_MODEL", "qwen2.5:7b"),
		OllamaURL:               getEnv("OLLAMA_URL", "http://host.docker.internal:11434"),
		LLMStatusTimeoutSeconds: getEnvInt("LLM_STATUS_TIMEOUT_SECONDS", 2),
	}

	if cfg.JWTSecret == "" {
		return nil, fmt.Errorf("JWT_SECRET environment variable is required")
	}

	return cfg, nil
}

func getEnv(key, fallback string) string {
	if value, ok := os.LookupEnv(key); ok {
		return value
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	value, ok := os.LookupEnv(key)
	if !ok {
		return fallback
	}
	var parsed int
	if _, err := fmt.Sscanf(value, "%d", &parsed); err != nil || parsed <= 0 {
		return fallback
	}
	return parsed
}
