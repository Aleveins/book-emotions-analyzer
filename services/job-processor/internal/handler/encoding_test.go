package handler

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToUTF8KeepsValidUTF8(t *testing.T) {
	input := []byte("Привет, мир")

	got, err := toUTF8(input)
	require.NoError(t, err)
	assert.Equal(t, input, got)
}

func TestToUTF8DecodesWindows1251(t *testing.T) {
	input := []byte{0xcf, 0xf0, 0xe8, 0xe2, 0xe5, 0xf2}

	got, err := toUTF8(input)
	require.NoError(t, err)
	assert.Equal(t, "Привет", string(got))
}
