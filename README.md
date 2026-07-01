# 🔬 ResumeAI — Semantic Resume Screening System

A two-tier AI-powered resume screening system built with **Streamlit** and **Google Gemini AI** for a Capstone Project (2026).

> **Traditional ATS use rigid keyword matching.** This system goes beyond keywords — using **Semantic Vector Embeddings** to understand context and meaning, not just exact matches.

---

## ✨ Features

### ⚡ BASIC Mode (Zero API Tokens)
- **TF-IDF Cosine Similarity** — Industry-standard keyword matching
- **Keyword Analysis** — Matched & missing JD keywords
- **Section Detection** — Education, Experience, Skills, Projects, etc.
- **Contact Extraction** — Email, phone, LinkedIn, GitHub
- **Composite ATS Score** — Weighted scoring (0-100%)
- **Candidate Leaderboard** — Ranked by composite score

### 🧠 PRO Mode (Gemini-Powered, On-Demand)
- **Semantic Match Score** — Vector embedding similarity via `gemini-embedding-001`
- **Contextual Skill Matching** — LLM identifies skills even when phrased differently
- **Critical Gap Analysis** — Missing skills that matter for the role
- **AI Fit Assessment** — Natural language candidate evaluation
- **Score Delta** — See how much semantic analysis changes rankings

---

## 🛠️ Tech Stack

| Component | Library | Purpose |
|:---|:---|:---|
| UI | `streamlit` | Interactive web interface |
| GenAI SDK | `google-genai` | Gemini API (unified SDK) |
| PDF Parsing | `PyMuPDF` | Fast PDF text extraction |
| DOCX Parsing | `python-docx` | Word document parsing |
| ML | `scikit-learn` | TF-IDF vectorization & cosine similarity |
| Math | `numpy` | Vector operations for embeddings |
| Data | `pandas` | Leaderboard DataFrames |

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/kravitexx/resume_screening_system.git
cd resume_screening_system
pip install -r requirements.txt
```

### 2. Configure API Key
```bash
# Edit .streamlit/secrets.toml with your Gemini API key
# Get your key from: https://aistudio.google.com/apikey
```

```toml
GEMINI_API_KEY = "your-actual-api-key-here"
```

### 3. Run
```bash
streamlit run app.py
```

> **Note:** BASIC mode works without an API key. PRO mode requires a valid Gemini API key.

---

## 📁 Project Structure

```
Resume Screening System/
├── .streamlit/
│   ├── config.toml          # Streamlit dark theme
│   └── secrets.toml         # API key (git-ignored)
├── assets/
│   └── style.css            # Apple glassmorphism CSS
├── utils/
│   ├── __init__.py
│   ├── text_extraction.py   # PDF/DOCX parsing
│   ├── basic_analyzer.py    # TF-IDF keyword analysis
│   ├── gemini_client.py     # Gemini client + retry + fallback
│   └── pro_analyzer.py      # Embeddings + LLM analysis
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

---

## 🔐 API Configuration

This project uses the **`google-genai`** SDK (the unified Google GenAI SDK — NOT the deprecated `google-generativeai`).

| Model | Purpose | Tier |
|:---|:---|:---|
| `gemini-2.0-flash` | LLM analysis (primary) | Free |
| `gemini-2.5-flash` | LLM analysis (fallback) | Free |
| `gemini-embedding-001` | Vector embeddings | Free |

---

## 📄 License

Capstone Project — 2026
