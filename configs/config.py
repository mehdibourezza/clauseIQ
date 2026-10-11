from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = "ClauseIQ"
    app_version: str = "1.0.0"
    app_env: str = "development"
    debug: bool = True

    # API
    api_prefix: str = "/v1"
    secret_key: str = "change-me-in-production"
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 days

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "clauseiq"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}"

    # Celery
    @property
    def celery_broker_url(self) -> str:
        return self.redis_url

    @property
    def celery_result_backend(self) -> str:
        return self.redis_url

    # Storage (S3-compatible)
    storage_bucket: str = "clauseiq-contracts"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_endpoint_url: str = ""  # empty = real AWS, set for MinIO

    # ML Model
    model_name: str = "nlpaueb/legal-bert-base-uncased"
    model_path: str = "ml/models/weights/clause_classifier"
    max_sequence_length: int = 512
    inference_batch_size: int = 8

    # Contract review app (LLM extraction engine)
    anthropic_api_key: str = ""          # empty = SDK reads ANTHROPIC_API_KEY from the environment
    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "high"             # low | medium | high | xhigh | max
    review_db_url: str = "sqlite:///clauseiq_review.db"

    # Rate limiting (requests per minute per API key)
    rate_limit_free: int = 10
    rate_limit_pro: int = 100
    rate_limit_enterprise: int = 1000

    # W&B
    wandb_api_key: str = ""
    wandb_project: str = "clauseiq"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
