package handler

import (
	"errors"
	"io"
	"log/slog"
	"net/http"
	"path/filepath"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/Aleveins/book-emotions-analyzer/services/job-processor/internal/service"
)

var supportedFormats = map[string]bool{
	"txt": true,
	"pdf": true,
}

func detectFormat(filename string) string {
	ext := strings.ToLower(strings.TrimPrefix(filepath.Ext(filename), "."))
	return ext
}

func parseFlag(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "on", "yes":
		return true
	default:
		return false
	}
}

type JobHandler struct {
	jobService       *service.JobService
	llmStatusService *service.LLMStatusService
}

func NewJobHandler(jobService *service.JobService, llmStatusService *service.LLMStatusService) *JobHandler {
	return &JobHandler{jobService: jobService, llmStatusService: llmStatusService}
}

func NewRouter(jobService *service.JobService, llmStatusService *service.LLMStatusService, jwtSecret string) *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Recovery())

	h := NewJobHandler(jobService, llmStatusService)

	v1 := router.Group("/api/v1")
	v1.Use(AuthMiddleware(jwtSecret))
	{
		v1.GET("/llm/status", h.GetLLMStatus)

		jobs := v1.Group("/jobs")
		{
			jobs.POST("", h.CreateJob)
			jobs.GET("", h.ListJobs)
			jobs.GET("/:id", h.GetJob)
			jobs.GET("/:id/result", h.GetResult)
		}
	}

	return router
}

func (h *JobHandler) CreateJob(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
		return
	}

	file, header, err := c.Request.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "file is required"})
		return
	}
	defer file.Close()

	format := detectFormat(header.Filename)
	if !supportedFormats[format] {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "unsupported file format: " + format + ". Supported: txt, pdf",
		})
		return
	}

	rawContent, err := io.ReadAll(file)
	if err != nil {
		slog.Error("failed to read file", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to read file"})
		return
	}

	// TXT-файлы могут приходить в старых кодировках, например Windows-1251
	// gRPC принимает произвольные bytes, но Python-обработчику удобнее получать
	// единый UTF-8
	if format == "txt" {
		converted, err := toUTF8(rawContent)
		if err != nil {
			slog.Error("failed to decode txt", "error", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": "unsupported text encoding"})
			return
		}
		rawContent = converted
	}

	title := c.PostForm("title")
	if title == "" {
		title = header.Filename
	}

	useLLM := parseFlag(c.PostForm("use_llm"))
	useParagraphAnalysis := parseFlag(c.PostForm("use_paragraph_analysis"))
	if useLLM {
		status := h.llmStatusService.Status(c.Request.Context())
		if !status.Available {
			c.JSON(http.StatusBadRequest, gin.H{"error": "дополнительная LLM-модель сейчас недоступна"})
			return
		}
	}

	job, err := h.jobService.CreateJob(c.Request.Context(), userID.(string), title, header.Filename, format, rawContent, useLLM, useParagraphAnalysis)
	if err != nil {
		slog.Error("failed to create job", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to create job"})
		return
	}

	c.JSON(http.StatusCreated, gin.H{"job_id": job.ID})
}

func (h *JobHandler) GetLLMStatus(c *gin.Context) {
	c.JSON(http.StatusOK, h.llmStatusService.Status(c.Request.Context()))
}

func (h *JobHandler) ListJobs(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
		return
	}

	jobs, err := h.jobService.ListJobs(c.Request.Context(), userID.(string))
	if err != nil {
		slog.Error("failed to list jobs", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to list jobs"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"jobs": jobs})
}

func (h *JobHandler) GetJob(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
		return
	}

	jobID := c.Param("id")

	job, err := h.jobService.GetJob(c.Request.Context(), jobID, userID.(string))
	if err != nil {
		if errors.Is(err, service.ErrJobNotFound) {
			c.JSON(http.StatusNotFound, gin.H{"error": "job not found"})
			return
		}
		slog.Error("failed to get job", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to get job"})
		return
	}

	c.JSON(http.StatusOK, job)
}

func (h *JobHandler) GetResult(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
		return
	}

	jobID := c.Param("id")

	result, err := h.jobService.GetResult(c.Request.Context(), jobID, userID.(string))
	if err != nil {
		if errors.Is(err, service.ErrJobNotFound) {
			c.JSON(http.StatusNotFound, gin.H{"error": "job not found"})
			return
		}
		if errors.Is(err, service.ErrJobNotComplete) {
			c.JSON(http.StatusBadRequest, gin.H{"error": "job is not completed yet"})
			return
		}
		slog.Error("failed to get result", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to get result"})
		return
	}

	c.JSON(http.StatusOK, result)
}
