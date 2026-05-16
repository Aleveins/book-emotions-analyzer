import os
from dataclasses import dataclass


@dataclass
class Config:
    nats_url: str
    redis_addr: str
    text_storage_addr: str
    model_name_or_path: str
    local_files_only: bool
    job_timeout_seconds: int
    llm_provider: str
    llm_model: str
    ollama_url: str
    llm_timeout_seconds: int

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            nats_url=os.getenv("NATS_URL", "nats://nats:4222"),
            redis_addr=os.getenv("REDIS_ADDR", "redis:6379"),
            text_storage_addr=os.getenv("TEXT_STORAGE_ADDR", "text-storage:50051"),
            model_name_or_path=os.getenv("MODEL_NAME_OR_PATH", "SamLowe/roberta-base-go_emotions"),
            local_files_only=os.getenv("LOCAL_FILES_ONLY", "false").lower() == "true",
            # 4 часа по умолчанию хватает для больших русских книг на CPU
            job_timeout_seconds=int(os.getenv("JOB_TIMEOUT_SECONDS", "14400")),
            llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
            llm_model=os.getenv("LLM_MODEL", "qwen2.5:7b"),
            ollama_url=os.getenv("OLLAMA_URL", "http://host.docker.internal:11434"),
            llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
        )
