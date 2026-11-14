"""Configuration definitions for LLM and agent execution."""

import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file in the workspace
load_dotenv()


def get_required_env_model() -> str:
    """Retrieves the Ollama model name strictly from the environment/.env.
    Defaults to 'qwen2.5-coder:3b' if not set.
    """
    model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")
    return model.strip()


def is_thinking_model_name(model_name: str) -> bool:
    """Auto-detects if the configured model is a reasoning/thinking model (e.g. R1)."""
    name = model_name.lower()
    return any(k in name for k in ("r1", "reason", "thinking", "deepseek-r"))


class LLMConfig(BaseModel):
    """Configuration for local Ollama LLM."""

    model: str = Field(
        default_factory=get_required_env_model,
        description="Ollama model tag loaded dynamically from OLLAMA_MODEL in .env",
    )
    base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        description="Local Ollama server API base URL",
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("OLLAMA_TEMPERATURE", "0.0")),
        ge=0.0,
        le=2.0,
        description="Sampling temperature for deterministic tool execution",
    )
    num_ctx: int = Field(
        default_factory=lambda: int(os.getenv("OLLAMA_NUM_CTX", "2048")),
        description="Ollama context window size (optimized to 2048 to maximize GPU offloading)",
    )
    max_retries: int = Field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")),
        ge=1,
        le=5,
        description="Maximum retry attempts for LLM generation and tool executions (limit 2-3)",
    )
    is_thinking_model: Optional[bool] = Field(
        default=None,
        description="Flag to activate thinking/reasoning model middleware (<think> tags). Auto-detected if None.",
    )

    def resolve_is_thinking_model(self) -> bool:
        if self.is_thinking_model is not None:
            return self.is_thinking_model
        env_val = os.getenv("IS_THINKING_MODEL")
        if env_val is not None:
            return env_val.strip().lower() in ("true", "1", "yes")
        return is_thinking_model_name(self.model)


class AgentSettings(BaseModel):
    """Overall settings for the manufacturing reasoning agent."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
