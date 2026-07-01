# ============================================================
# basic_analyzer.py — BASIC Mode Analysis (Zero Gemini Tokens)
# ============================================================
# All computations are local using scikit-learn's TF-IDF and
# regex-based pattern matching. No API calls required.
#
# Metrics computed:
#   - TF-IDF Cosine Similarity (ATS Keyword Score)
#   - Keyword matching (matched + missing)
#   - Resume section detection (Education, Experience, etc.)
#   - Contact information extraction (email, phone, LinkedIn)
#   - Word count and estimated page count
# ============================================================

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---- Common English Stop Words (lightweight, no NLTK dependency) ----
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "must",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "them", "his", "her", "its", "their", "this", "that", "these", "those",
    "am", "not", "no", "nor", "so", "if", "then", "than", "too", "very",
    "just", "about", "above", "after", "again", "all", "also", "any",
    "because", "before", "below", "between", "both", "each", "few", "more",
    "most", "other", "own", "same", "some", "such", "up", "down", "out",
    "over", "under", "here", "there", "when", "where", "which", "while",
    "who", "whom", "what", "how", "as", "into", "through", "during",
    "etc", "via", "per", "upon",
}

# ---- Resume Section Patterns ----
# Common section headers found in resumes, with regex patterns
SECTION_PATTERNS = {
    "Education": r"\b(education|academic|qualification|degree|university|college|school|gpa|cgpa|bachelor|master|phd|diploma)\b",
    "Experience": r"\b(experience|employment|work\s*history|professional\s*experience|career|internship|intern)\b",
    "Skills": r"\b(skills|technical\s*skills|core\s*competencies|technologies|proficiencies|tools|frameworks)\b",
    "Projects": r"\b(projects|personal\s*projects|academic\s*projects|portfolio)\b",
    "Certifications": r"\b(certification|certifications|certified|certificate|licenses|accreditation)\b",
    "Summary": r"\b(summary|objective|profile|about\s*me|professional\s*summary|career\s*objective)\b",
    "Awards": r"\b(awards|achievements|honors|recognition|accomplishments)\b",
    "Publications": r"\b(publications|papers|research|journal|conference)\b",
    "Languages": r"\b(languages|language\s*proficiency|fluent|native)\b",
    "References": r"\b(references|referees|available\s*upon\s*request)\b",
}

# ---- Contact Info Patterns ----
EMAIL_PATTERN = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
PHONE_PATTERN = r"(?:\+?\d{1,3}[\s\-]?)?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}"
LINKEDIN_PATTERN = r"(?:linkedin\.com/in/[\w\-]+|linkedin\.com/[\w\-]+)"
GITHUB_PATTERN = r"(?:github\.com/[\w\-]+)"


def preprocess_text(text: str) -> str:
    """
    Clean and normalize text for analysis.

    Steps:
    1. Convert to lowercase
    2. Remove special characters (keep alphanumeric and spaces)
    3. Remove extra whitespace
    4. Remove stop words

    Args:
        text: Raw text string.

    Returns:
        Cleaned, normalized text.
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses (preserve them separately for contact extraction)
    text = re.sub(EMAIL_PATTERN, " ", text)

    # Remove special characters but keep alphanumeric, spaces, and hyphens
    text = re.sub(r"[^a-z0-9\s\-+#.]", " ", text)

    # Remove standalone single characters
    text = re.sub(r"\b[a-z]\b", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stop words
    words = text.split()
    filtered_words = [w for w in words if w not in STOP_WORDS and len(w) > 1]

    return " ".join(filtered_words)


def compute_tfidf_score(jd_text: str, resume_text: str) -> float:
    """
    Compute TF-IDF cosine similarity between job description and resume.

    This is the core ATS keyword matching score. TF-IDF weights words
    by their importance (penalizing common words, rewarding rare ones),
    and cosine similarity measures how aligned the two documents are.

    Args:
        jd_text: Job description text.
        resume_text: Resume text.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    if not jd_text or not resume_text:
        return 0.0

    # Preprocess both texts
    clean_jd = preprocess_text(jd_text)
    clean_resume = preprocess_text(resume_text)

    if not clean_jd or not clean_resume:
        return 0.0

    try:
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),  # Unigrams and bigrams for phrase matching
        )
        tfidf_matrix = vectorizer.fit_transform([clean_jd, clean_resume])

        # Cosine similarity between JD vector and resume vector
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        return float(max(0.0, min(1.0, score)))

    except Exception:
        return 0.0


def extract_keywords(text: str, top_n: int = 30) -> list[str]:
    """
    Extract the most important keywords from text using TF-IDF.

    Identifies the top N words/phrases that best characterize the document.

    Args:
        text: Input text.
        top_n: Number of top keywords to extract.

    Returns:
        List of keyword strings, sorted by TF-IDF importance.
    """
    if not text:
        return []

    clean_text = preprocess_text(text)
    if not clean_text:
        return []

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=200,
            ngram_range=(1, 2),
        )
        tfidf_matrix = vectorizer.fit_transform([clean_text])

        # Get feature names and their TF-IDF scores
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]

        # Sort by score descending and return top_n
        keyword_scores = sorted(
            zip(feature_names, scores), key=lambda x: x[1], reverse=True
        )

        return [kw for kw, score in keyword_scores[:top_n] if score > 0]

    except Exception:
        return []


def find_matched_missing_keywords(
    jd_text: str, resume_text: str, top_n: int = 25
) -> tuple[list[str], list[str]]:
    """
    Find which JD keywords are present in (matched) or absent from
    (missing) the resume.

    Uses TF-IDF to extract important JD keywords, then checks each
    against the resume text using word-level matching.

    Args:
        jd_text: Job description text.
        resume_text: Resume text.
        top_n: Number of JD keywords to check.

    Returns:
        Tuple of (matched_keywords, missing_keywords).
    """
    jd_keywords = extract_keywords(jd_text, top_n=top_n)

    if not jd_keywords:
        return [], []

    resume_lower = preprocess_text(resume_text)

    matched = []
    missing = []

    for keyword in jd_keywords:
        # Check if the keyword (or close variant) appears in the resume
        if keyword.lower() in resume_lower:
            matched.append(keyword)
        else:
            # Also check individual words for multi-word keywords
            words = keyword.lower().split()
            if len(words) > 1 and all(w in resume_lower for w in words):
                matched.append(keyword)
            else:
                missing.append(keyword)

    return matched, missing


def detect_sections(resume_text: str) -> dict[str, bool]:
    """
    Detect which standard resume sections are present.

    Checks for common section headers (Education, Experience, Skills, etc.)
    using regex pattern matching.

    Args:
        resume_text: Resume text.

    Returns:
        Dictionary mapping section name → boolean (found or not).
    """
    if not resume_text:
        return {section: False for section in SECTION_PATTERNS}

    text_lower = resume_text.lower()
    results = {}

    for section_name, pattern in SECTION_PATTERNS.items():
        results[section_name] = bool(re.search(pattern, text_lower))

    return results


def extract_contact_info(resume_text: str) -> dict[str, str | None]:
    """
    Extract contact information from resume text.

    Looks for email addresses, phone numbers, LinkedIn profiles,
    and GitHub profiles using regex patterns.

    Args:
        resume_text: Resume text.

    Returns:
        Dictionary with keys: email, phone, linkedin, github.
        Values are the first match found, or None.
    """
    if not resume_text:
        return {"email": None, "phone": None, "linkedin": None, "github": None}

    # Extract email
    email_match = re.search(EMAIL_PATTERN, resume_text, re.IGNORECASE)
    email = email_match.group(0) if email_match else None

    # Extract phone
    phone_match = re.search(PHONE_PATTERN, resume_text)
    phone = phone_match.group(0).strip() if phone_match else None

    # Extract LinkedIn
    linkedin_match = re.search(LINKEDIN_PATTERN, resume_text, re.IGNORECASE)
    linkedin = linkedin_match.group(0) if linkedin_match else None

    # Extract GitHub
    github_match = re.search(GITHUB_PATTERN, resume_text, re.IGNORECASE)
    github = github_match.group(0) if github_match else None

    return {
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "github": github,
    }


def compute_section_score(sections: dict[str, bool]) -> float:
    """
    Calculate a section completeness score.

    Essential sections (Education, Experience, Skills) are weighted higher.

    Args:
        sections: Output from detect_sections().

    Returns:
        Score between 0.0 and 1.0.
    """
    # Define weights for each section
    weights = {
        "Education": 3.0,
        "Experience": 3.0,
        "Skills": 3.0,
        "Projects": 2.0,
        "Summary": 1.5,
        "Certifications": 1.0,
        "Awards": 0.5,
        "Publications": 0.5,
        "Languages": 0.5,
        "References": 0.5,
    }

    total_weight = sum(weights.get(s, 1.0) for s in sections)
    earned_weight = sum(
        weights.get(s, 1.0) for s, found in sections.items() if found
    )

    return earned_weight / total_weight if total_weight > 0 else 0.0


def get_word_count(text: str) -> int:
    """Get the word count of a text string."""
    return len(text.split()) if text else 0


def estimate_pages(text: str) -> int:
    """Estimate page count (roughly 400-500 words per page)."""
    word_count = get_word_count(text)
    return max(1, round(word_count / 450))


def analyze_resume_basic(jd_text: str, resume_text: str, file_name: str) -> dict:
    """
    Run the complete BASIC analysis pipeline on a single resume.

    This is the main entry point for BASIC mode. It computes ALL local
    metrics without making any API calls.

    Args:
        jd_text: Job description text.
        resume_text: Extracted resume text.
        file_name: Original filename for display.

    Returns:
        Dictionary containing all BASIC analysis results:
        - file_name: Original filename
        - ats_score: TF-IDF cosine similarity (0-100)
        - matched_keywords: List of matched JD keywords
        - missing_keywords: List of missing JD keywords
        - sections: Dict of detected sections
        - section_score: Section completeness (0-100)
        - contact_info: Dict of extracted contact details
        - word_count: Total word count
        - estimated_pages: Estimated page count
        - composite_score: Weighted overall score (0-100)
    """
    # Core TF-IDF score
    tfidf_score = compute_tfidf_score(jd_text, resume_text)

    # Keyword analysis
    matched, missing = find_matched_missing_keywords(jd_text, resume_text)

    # Section detection
    sections = detect_sections(resume_text)
    section_score = compute_section_score(sections)

    # Contact info
    contact_info = extract_contact_info(resume_text)

    # Text metrics
    word_count = get_word_count(resume_text)
    pages = estimate_pages(resume_text)

    # Compute keyword match ratio
    total_keywords = len(matched) + len(missing)
    keyword_ratio = len(matched) / total_keywords if total_keywords > 0 else 0.0

    # ---- Composite Score (weighted combination) ----
    # Weights:
    #   - TF-IDF Score:     40% (core keyword relevance)
    #   - Keyword Match:    35% (specific keyword coverage)
    #   - Section Score:    25% (resume completeness)
    composite = (tfidf_score * 0.40) + (keyword_ratio * 0.35) + (section_score * 0.25)

    return {
        "file_name": file_name,
        "ats_score": round(tfidf_score * 100, 1),
        "matched_keywords": matched,
        "missing_keywords": missing,
        "keyword_match_count": len(matched),
        "keyword_total": total_keywords,
        "sections": sections,
        "section_score": round(section_score * 100, 1),
        "contact_info": contact_info,
        "word_count": word_count,
        "estimated_pages": pages,
        "composite_score": round(composite * 100, 1),
    }
