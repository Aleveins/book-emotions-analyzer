package handler

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Aleveins/book-emotions-analyzer/pkg/jwt"
)

func TestAuthMiddlewareAcceptsValidBearerToken(t *testing.T) {
	gin.SetMode(gin.TestMode)

	const secret = "test-secret"
	token, err := jwt.GenerateToken("user-1", "ivan", secret, time.Hour)
	require.NoError(t, err)

	router := gin.New()
	router.GET("/me", AuthMiddleware(secret), func(c *gin.Context) {
		assert.Equal(t, "user-1", c.GetString("user_id"))
		assert.Equal(t, "ivan", c.GetString("login"))
		c.Status(http.StatusNoContent)
	})

	req := httptest.NewRequest(http.MethodGet, "/me", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	assert.Equal(t, http.StatusNoContent, resp.Code, resp.Body.String())
}

func TestAuthMiddlewareRejectsMissingBearerToken(t *testing.T) {
	gin.SetMode(gin.TestMode)

	called := false
	router := gin.New()
	router.GET("/me", AuthMiddleware("test-secret"), func(c *gin.Context) {
		called = true
	})

	req := httptest.NewRequest(http.MethodGet, "/me", nil)
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	assert.Equal(t, http.StatusUnauthorized, resp.Code)
	assert.False(t, called)
}
