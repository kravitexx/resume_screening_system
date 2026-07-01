# ============================================================
# pro_analyzer.py — PRO Mode Analysis (Gemini-Powered)
# ============================================================
# Uses Gemini API for:
#   1. Vector Embeddings (gemini-embedding-001) → Semantic score
#   2. LLM Contextual Extraction → Matched/Missing skills + assessment
#
# This module is ONLY called when the user clicks "Try PRO".
# ============================================================

import json
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from google import genai

from utils.gemini_client import (
    generate_embeddings_with_client,
    generate_content_with_fallback,
)


def compute_semantic_score(
    client: genai.Client, jd_text: str, resume_text: str
) -> float | None:
    """
    Compute semantic similarity between JD and resume using Gemini embeddings.

    Uses gemini-embedding-001 with task_type optimization:
    - JD is embedded as RETRIEVAL_QUERY (what we're searching for)
    - Resume is embedded as RETRIEVAL_DOCUMENT (what we're searching in)

    This asymmetric embedding approach significantly improves retrieval quality
    compared to embedding both with the same task type.

    Args:
        client: Initialized genai.Client.
        jd_text: Job description text.
        resume_text: Resume text.

    Returns:
        Similarity score (0.0 to 1.0), or None on failure.
    """
    # Truncate very long texts to stay within token limits
    # gemini-embedding-001 handles up to ~2048 tokens well
    jd_truncated = jd_text[:8000]
    resume_truncated = resume_text[:8000]

    # Generate JD embedding with RETRIEVAL_QUERY task type
    jd_embeddings = generate_embeddings_with_client(
        client, [jd_truncated], task_type="RETRIEVAL_QUERY"
    )
    if not jd_embeddings:
        return None

    # Generate resume embedding with RETRIEVAL_DOCUMENT task type
    resume_embeddings = generate_embeddings_with_client(
        client, [resume_truncated], task_type="RETRIEVAL_DOCUMENT"
    )
    if not resume_embeddings:
        return None

    # Compute cosine similarity using numpy
    jd_vec = np.array(jd_embeddings[0]).reshape(1, -1)
    resume_vec = np.array(resume_embeddings[0]).reshape(1, -1)

    similarity = sklearn_cosine(jd_vec, resume_vec)[0][0]

    # Clamp to [0, 1] range
    return float(max(0.0, min(1.0, similarity)))


def extract_skills_with_llm(
    client: genai.Client, jd_text: str, resume_text: str, job_type: str = "General"
) -> dict | None:
    """
    Use Gemini LLM to perform contextual skill extraction and gap analysis.

    Sends the JD and resume to the model with a structured prompt,
    requesting JSON output with:
    - Contextually matched skills (semantically, not just keyword match)
    - Critical missing skills
    - Overall fit assessment

    Args:
        client: Initialized genai.Client.
        jd_text: Job description text.
        resume_text: Resume text.
        job_type: Detected or user-specified job type.

    Returns:
        Dictionary with keys:
        - matched_skills: List of contextually matched skill strings
        - missing_skills: List of critical missing skill strings
        - fit_assessment: Brief text assessment
        - fit_level: "Strong Fit" | "Moderate Fit" | "Weak Fit"
        Returns None on complete failure.
    """
    prompt = f"""You are an expert technical recruiter and ATS specialist analyzing a candidate's resume against a job description.

**Job Type:** {job_type}

**Job Description:**
{jd_text[:4000]}

**Candidate's Resume:**
{resume_text[:6000]}

**Your Task:**
Analyze the resume against the job description and return a JSON object with the following structure. Be thorough and consider SEMANTIC matches — not just exact keyword matches.

For example:
- "React.js" in the resume matches "Frontend development" in the JD
- "Led team of 5" matches "Team leadership" requirement
- "Python, TensorFlow" matches "Machine Learning experience"

Return ONLY valid JSON in this exact format — no markdown, no code fences, no explanation:
{{
    "matched_skills": ["skill1", "skill2", "skill3"],
    "missing_skills": ["skill1", "skill2"],
    "fit_assessment": "A 2-3 sentence assessment of the candidate's overall fit for this role.",
    "fit_level": "Strong Fit"
}}

Rules for fit_level:
- "Strong Fit": 70%+ of key requirements matched
- "Moderate Fit": 40-70% of key requirements matched  
- "Weak Fit": Below 40% of key requirements matched

Rules for matched_skills:
- Include skills that are contextually relevant, even if phrased differently
- Be specific (e.g., "Python programming" not just "programming")
- Include both hard skills and relevant soft skills

Rules for missing_skills:
- Only include skills that are CRITICAL for the role
- Don't list nice-to-haves as missing
- Be specific about what's missing"""

    response_text = generate_content_with_fallback(client, prompt)

    if not response_text:
        return None

    return _parse_llm_response(response_text)


def _parse_llm_response(response_text: str) -> dict | None:
    """
    Parse the LLM's JSON response with fallback regex extraction.

    The model sometimes wraps JSON in markdown code fences or adds
    extra text. This function handles those cases gracefully.

    Args:
        response_text: Raw text response from the LLM.

    Returns:
        Parsed dictionary, or None if parsing fails completely.
    """
    # Default structure
    default = {
        "matched_skills": [],
        "missing_skills": [],
        "fit_assessment": "Unable to parse LLM response.",
        "fit_level": "Unknown",
    }

    if not response_text:
        return default

    # Attempt 1: Direct JSON parse
    try:
        data = json.loads(response_text.strip())
        return _validate_parsed_data(data, default)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Extract JSON from markdown code fences
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return _validate_parsed_data(data, default)
        except json.JSONDecodeError:
            pass

    # Attempt 3: Find the first { ... } block in the response
    brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response_text, re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group(0))
            return _validate_parsed_data(data, default)
        except json.JSONDecodeError:
            pass

    # Attempt 4: Regex-based extraction as last resort
    return _regex_fallback_extraction(response_text, default)


def _validate_parsed_data(data: dict, default: dict) -> dict:
    """Validate and normalize the parsed JSON structure."""
    result = default.copy()

    if isinstance(data.get("matched_skills"), list):
        result["matched_skills"] = [str(s) for s in data["matched_skills"]]

    if isinstance(data.get("missing_skills"), list):
        result["missing_skills"] = [str(s) for s in data["missing_skills"]]

    if isinstance(data.get("fit_assessment"), str):
        result["fit_assessment"] = data["fit_assessment"]

    if isinstance(data.get("fit_level"), str) and data["fit_level"] in {
        "Strong Fit",
        "Moderate Fit",
        "Weak Fit",
    }:
        result["fit_level"] = data["fit_level"]

    return result


def _regex_fallback_extraction(response_text: str, default: dict) -> dict:
    """
    Last-resort regex extraction when JSON parsing fails completely.
    Tries to extract skill lists from the raw text.
    """
    result = default.copy()

    # Try to extract lists of skills from patterns like "- skill" or "* skill"
    skills_found = re.findall(r"[-*•]\s*(.+?)(?:\n|$)", response_text)
    if skills_found:
        # Rough heuristic: first half are matched, second half are missing
        mid = len(skills_found) // 2
        result["matched_skills"] = [s.strip() for s in skills_found[:mid]]
        result["missing_skills"] = [s.strip() for s in skills_found[mid:]]
        result["fit_assessment"] = "Extracted via fallback parsing."

    return result


def analyze_resume_pro(
    client: genai.Client,
    jd_text: str,
    resume_text: str,
    file_name: str,
    job_type: str = "General",
) -> dict:
    """
    Run the complete PRO analysis pipeline on a single resume.

    This is the main entry point for PRO mode. It generates embeddings
    and runs LLM contextual extraction.

    Args:
        client: Initialized genai.Client.
        jd_text: Job description text.
        resume_text: Resume text.
        file_name: Original filename.
        job_type: Job type for context.

    Returns:
        Dictionary containing all PRO analysis results:
        - file_name: Original filename
        - semantic_score: Embedding cosine similarity (0-100)
        - matched_skills: Contextually matched skills
        - missing_skills: Critical missing skills
        - fit_assessment: LLM assessment text
        - fit_level: Strong/Moderate/Weak Fit
    """
    # Step 1: Semantic similarity via embeddings
    semantic_score = compute_semantic_score(client, jd_text, resume_text)

    # Step 2: LLM skill extraction
    llm_result = extract_skills_with_llm(client, jd_text, resume_text, job_type)

    # Build result
    result = {
        "file_name": file_name,
        "semantic_score": round(semantic_score * 100, 1) if semantic_score is not None else 0.0,
        "matched_skills": llm_result.get("matched_skills", []) if llm_result else [],
        "missing_skills": llm_result.get("missing_skills", []) if llm_result else [],
        "fit_assessment": llm_result.get("fit_assessment", "Analysis unavailable.") if llm_result else "Analysis unavailable.",
        "fit_level": llm_result.get("fit_level", "Unknown") if llm_result else "Unknown",
    }

    return result
