# ============================================================
# gemini_client.py — Gemini API Client with Retry & Fallback
# ============================================================
# Initializes the google-genai client with:
#   - Built-in retry (HttpRetryOptions) for 429/5xx errors
#   - Model fallback chain: gemini-2.0-flash → gemini-2.5-flash
#   - Secure API key loading from st.secrets
# ============================================================

import streamlit as st
from google import genai
from google.genai import types


# ---- Model Configuration ----
# Primary and fallback models for the free tier.
# To upgrade to Gemini 3.5 Flash, change PRIMARY_MODEL below.
PRIMARY_MODEL = "gemini-2.5-flash-lite"
FALLBACK_MODEL = "gemini-2.5-flash"

# Embedding model — gemini-embedding-001 replaced deprecated text-embedding-004
EMBEDDING_MODEL = "gemini-embedding-001"


def init_client() -> genai.Client | None:
    """
    Initialize the Gemini GenAI client with retry configuration.

    Reads the API key from Streamlit secrets and configures:
    - 10 retry attempts with exponential backoff
    - Retries on: 408, 429, 500, 502, 503, 504
    - 120-second timeout per request

    Returns:
        genai.Client instance, or None if API key is missing/invalid.
    """
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)

        if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
            return None

        # Configure retry with exponential backoff for rate limits
        # Using 3 attempts with 60s timeout to avoid long hangs
        retry_config = types.HttpRetryOptions(
            initial_delay=1.0,
            attempts=3,
            http_status_codes=[408, 429, 500, 502, 503, 504],
        )

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                retry_options=retry_config,
                timeout=60 * 1000,  # 60 seconds in milliseconds
            ),
        )

        return client

    except Exception as e:
        st.error(f"❌ Failed to initialize Gemini client: {e}")
        return None


def generate_content_with_fallback(
    client: genai.Client,
    prompt: str,
    primary_model: str = PRIMARY_MODEL,
    fallback_model: str = FALLBACK_MODEL,
) -> str | None:
    """
    Generate content with automatic model fallback.

    Tries the primary model first. If it fails (rate limit, quota,
    model unavailable), falls back to the secondary model.

    Args:
        client: Initialized genai.Client.
        prompt: The text prompt to send.
        primary_model: First model to try (default: gemini-2.0-flash).
        fallback_model: Backup model (default: gemini-2.5-flash).

    Returns:
        Generated text string, or None on complete failure.
    """
    # Try primary model
    try:
        response = client.models.generate_content(
            model=primary_model,
            contents=prompt,
        )
        return response.text

    except Exception as primary_error:
        st.warning(
            f"⚠️ Primary model ({primary_model}) failed: {primary_error}. "
            f"Trying fallback ({fallback_model})..."
        )

    # Try fallback model
    try:
        response = client.models.generate_content(
            model=fallback_model,
            contents=prompt,
        )
        return response.text

    except Exception as fallback_error:
        st.error(
            f"❌ Both models failed.\n"
            f"Primary ({primary_model}): {primary_error}\n"
            f"Fallback ({fallback_model}): {fallback_error}"
        )
        return None


def generate_embeddings_with_client(
    client: genai.Client,
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]] | None:
    """
    Generate vector embeddings for a list of texts using Gemini.

    Uses gemini-embedding-001 (3072 dimensions by default).
    Supports task_type for improved semantic quality:
      - "RETRIEVAL_DOCUMENT" for resumes/documents
      - "RETRIEVAL_QUERY" for job descriptions/queries

    Args:
        client: Initialized genai.Client.
        texts: List of text strings to embed.
        task_type: The embedding task type for optimized retrieval.

    Returns:
        List of embedding vectors (each a list of floats),
        or None on failure.
    """
    try:
        # Build the embedding config with task type
        config = types.EmbedContentConfig(task_type=task_type)

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=config,
        )

        # Extract the embedding values from the response
        embeddings = [embedding.values for embedding in response.embeddings]
        return embeddings

    except Exception as e:
        st.error(f"❌ Embedding generation failed: {e}")
        return None


def auto_detect_job_type(client: genai.Client, jd_text: str) -> str:
    """
    Use Gemini to automatically classify the job type from a job description.

    This is the ONLY Gemini usage in BASIC mode — called only when
    the user hasn't manually specified a job type.

    Args:
        client: Initialized genai.Client.
        jd_text: The job description text.

    Returns:
        Detected job type string (e.g., "Data Scientist", "Backend Engineer").
        Returns "General" on failure.
    """
    prompt = f"""Analyze the following job description and identify the specific job title/type.
Return ONLY the job title (e.g., "Data Scientist", "Frontend Developer", "Product Manager").
Do not include any explanation or additional text. Just the job title.

Job Description:
{jd_text[:2000]}"""

    result = generate_content_with_fallback(client, prompt)

    if result:
        # Clean up the response — remove quotes, newlines, extra whitespace
        cleaned = result.strip().strip('"').strip("'").strip()
        # Take only the first line if multiple were returned
        return cleaned.split("\n")[0].strip() if cleaned else "General"

    return "General"
