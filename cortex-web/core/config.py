"""Settings from environment variables, read when first needed (never at import time)."""
import os


class ConfigError(RuntimeError):
    """A required environment variable is missing."""


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"Server is not configured: {name} is not set")
    return value


def optional(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def database_url() -> str:
    return required("DATABASE_URL")


def supabase_url() -> str:
    return required("SUPABASE_URL").rstrip("/")


def supabase_secret_key() -> str:
    return required("SUPABASE_SECRET_KEY")


def gemini_api_key() -> str:
    return required("GEMINI_API_KEY")


MIN_PASSWORD_LEN = 12


def upload_password() -> str:
    value = required("UPLOAD_PASSWORD")
    if len(value) < MIN_PASSWORD_LEN:
        raise ConfigError(f"Server is not configured: UPLOAD_PASSWORD must have at least "
                          f"{MIN_PASSWORD_LEN} characters")
    return value


def session_secret() -> str:
    return required("SESSION_SECRET")


def gemini_embed_model() -> str:
    return optional("GEMINI_EMBED_MODEL", "gemini-embedding-001")


def storage_bucket() -> str:
    return optional("STORAGE_BUCKET", "cortex-docs")


def max_upload_bytes() -> int:
    return int(float(optional("MAX_UPLOAD_MB", "50")) * 1024 * 1024)


def ingest_budget_s() -> float:
    # Work time per /api/ingest call; Vercel Hobby kills the function at 300 s.
    return float(optional("INGEST_BUDGET_S", "240"))
