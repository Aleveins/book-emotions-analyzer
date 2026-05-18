package handler

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDetectFormat(t *testing.T) {
	tests := map[string]string{
		"book.TXT":       "txt",
		"archive.Pdf":    "pdf",
		"without_ext":    "",
		"chapter.v1.txt": "txt",
	}

	for filename, want := range tests {
		assert.Equal(t, want, detectFormat(filename), filename)
	}
}

func TestParseFlag(t *testing.T) {
	trueValues := []string{"1", "true", "TRUE", "on", " yes "}
	for _, value := range trueValues {
		assert.True(t, parseFlag(value), value)
	}

	falseValues := []string{"", "0", "false", "no", "anything"}
	for _, value := range falseValues {
		assert.False(t, parseFlag(value), value)
	}
}
