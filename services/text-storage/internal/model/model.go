package model

import "time"

type Text struct {
	ID        string
	UserID    string
	Title     string
	Filename  string
	Content   []byte
	Format    string
	CreatedAt time.Time
}

type AnalysisResult struct {
	ID         string
	TextID     string
	JobID      string
	ResultJSON []byte
	CreatedAt  time.Time
}
