"""
SentinelEdge — Backend Configuration
======================================
All configuration loaded from environment variables (.env file).
Using pydantic-settings for validation and type safety.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── Supabase ────────────────────────────────────────────────────────────
    supabase_url:         str = Field(..., env="SUPABASE_URL")
    supabase_service_key: str = Field(..., env="SUPABASE_SERVICE_KEY")

    # ── Groq LLM ────────────────────────────────────────────────────────────
    groq_api_key:  str = Field(..., env="GROQ_API_KEY")
    groq_model:    str = Field("llama-3.3-70b-versatile", env="GROQ_MODEL")
    groq_max_tokens: int = Field(200, env="GROQ_MAX_TOKENS")

    # ── Telegram ─────────────────────────────────────────────────────────────
    telegram_bot_token: str = Field(..., env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id:   str = Field(..., env="TELEGRAM_CHAT_ID")

    # ── MQTT ─────────────────────────────────────────────────────────────────
    mqtt_broker_host: str = Field("localhost", env="MQTT_BROKER_HOST")
    mqtt_broker_port: int = Field(1883,        env="MQTT_BROKER_PORT")
    mqtt_client_id:   str = Field("sentinel-backend", env="MQTT_CLIENT_ID")

    # ── Device ───────────────────────────────────────────────────────────────
    device_id: str = Field("sentineledge-001", env="DEVICE_ID")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton settings instance
settings = Settings()
