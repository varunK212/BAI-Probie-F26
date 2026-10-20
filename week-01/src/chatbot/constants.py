"""
Constants Module

This module contains application-level constants that define the chatbot's
behavior. These values are hard-coded because they're part of the application
logic, not environment-specific configuration.

Key Concepts:
- Constants: Values that don't change during execution
- Hard-coding: Putting values directly in code (appropriate for non-secrets)
- Type annotations: UPPER_CASE names indicate these are constants

The difference between constants and environment variables:
- Environment variables (.env): Secrets and deployment-specific values (API keys, database URLs)
- Constants (this file): Application behavior that's the same everywhere (model names, prompts)
"""

from typing import Literal

# OpenRouter uses provider/model identifiers for chat completions.
# Using Literal type to enforce type checking at development time
ValidModel = Literal["openai/gpt-4o-mini",]

# OpenRouter's OpenAI-compatible API endpoint and the model used by default.
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
DEFAULT_MODEL: ValidModel = "openai/gpt-4o-mini"

# The system prompt defines the chatbot's personality and behavior
# This message is sent to the AI at the start of every conversation
SYSTEM_PROMPT: str = """You are a helpful, friendly AI assistant. 

Your role is to:
- Answer questions clearly and accurately
- Be patient and encouraging, especially with students learning new concepts
- Explain complex topics in simple terms
- Admit when you don't know something rather than making things up

Keep your responses concise but informative."""
