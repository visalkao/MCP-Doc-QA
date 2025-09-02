from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "gpt-4o-mini"
    UPLOAD_DIR: str = "./data/uploads"
    INDEX_DIR: str = "./data/index"
    TOP_K: int = 5

    class Config:
        env_file = ".env"

settings = Settings()
