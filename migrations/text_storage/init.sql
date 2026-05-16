CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS texts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    title VARCHAR(512) NOT NULL,
    filename VARCHAR(512) NOT NULL,
    content BYTEA NOT NULL,
    format VARCHAR(16) NOT NULL DEFAULT 'txt',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_texts_user_id ON texts(user_id);

CREATE TABLE IF NOT EXISTS analysis_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    text_id UUID NOT NULL REFERENCES texts(id) ON DELETE CASCADE,
    job_id UUID NOT NULL UNIQUE,
    result_json JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_analysis_results_job_id ON analysis_results(job_id);
CREATE INDEX idx_analysis_results_text_id ON analysis_results(text_id);
