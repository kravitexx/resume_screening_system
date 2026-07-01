import json
import re
from google import genai
from utils.gemini_client import generate_content_with_fallback

def compare_candidates_pro(
    client: genai.Client,
    jd_text: str,
    resumes: dict[str, str],
    job_type: str = "General"
) -> str:
    """
    Use Gemini LLM to compare up to 4 candidates against the job description.
    
    Args:
        client: Initialized genai.Client.
        jd_text: Job description text.
        resumes: Dictionary of {file_name: resume_text}.
        job_type: Detected or user-specified job type.
        
    Returns:
        Markdown string containing the detailed comparison and ranking.
    """
    
    # Cap at 4 resumes to ensure we don't exceed token limits or confuse the model
    resume_items = list(resumes.items())[:4]
    
    # Build the prompt
    candidates_section = ""
    for i, (fname, rtext) in enumerate(resume_items):
        # Truncate each resume slightly to fit well within context
        candidates_section += f"### Candidate {i+1}: {fname}\n{rtext[:4000]}\n\n"
        
    prompt = f"""You are an expert Technical Recruiter and Hiring Manager.
You have been asked to review and compare the following candidates for this role.

**Job Type:** {job_type}

**Job Description:**
{jd_text[:3000]}

**Candidates:**
{candidates_section}

**Your Task:**
1. Compare these candidates based on their semantic fit, skills, and experience relevant to the Job Description.
2. Provide a clear ranking of the candidates from best fit to worst fit.
3. For each candidate, provide a brief summary of why they were ranked where they are (Strengths vs Weaknesses).
4. Conclude with a definitive recommendation on who to hire and why.

Format your response in clean Markdown. Use headings, bullet points, and bold text for readability. Do not wrap your entire response in a markdown code block.
"""

    response_text = generate_content_with_fallback(client, prompt)
    
    if not response_text:
        return "❌ Failed to generate comparison. The AI service may be unavailable or rate limited."
        
    return response_text
