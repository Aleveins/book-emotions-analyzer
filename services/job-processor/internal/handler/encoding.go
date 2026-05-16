package handler

import (
	"bytes"
	"io"
	"unicode/utf8"

	"golang.org/x/text/encoding/charmap"
	"golang.org/x/text/transform"
)

// toUTF8 возвращает содержимое без изменений, если вход уже в UTF-8; 
// иначе байты декодируются из Windows-1251, самой частой старой кодировки для ру текста
func toUTF8(content []byte) ([]byte, error) {
	if utf8.Valid(content) {
		return content, nil
	}
	r := transform.NewReader(bytes.NewReader(content), charmap.Windows1251.NewDecoder())
	return io.ReadAll(r)
}
