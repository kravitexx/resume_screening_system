# ============================================================
# app.py — Semantic Resume Screening System
# ============================================================
# A two-tier resume screening application:
#   BASIC Mode: TF-IDF keyword matching (zero API tokens)
#   PRO Mode:   Gemini embeddings + LLM analysis (on-demand)
#
# Tech Stack:
#   - Streamlit (UI)
#   - google-genai (Gemini SDK — NOT the deprecated library)
#   - PyMuPDF + python-docx (document parsing)
#   - scikit-learn (TF-IDF, cosine similarity)
#   - numpy, pandas (data processing)
#
# Architecture:
#   - BASIC runs immediately on "Analyze" click
#   - PRO only activates when user clicks "Try PRO"
#   - Gemini tokens are conserved by this lazy approach
# ============================================================

import streamlit as st
import pandas as pd
import os

# Import our utility modules
from utils.text_extraction import extract_text
from utils.basic_analyzer import analyze_resume_basic
from utils.gemini_client import init_client, auto_detect_job_type
from utils.pro_analyzer import analyze_resume_pro


# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="ResumeAI — Semantic Screening",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Load Custom CSS (Apple Glassmorphism)
# ============================================================
def load_css():
    """Inject the glassmorphism CSS stylesheet."""
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.html(f"<style>{f.read()}</style>")


load_css()


# ============================================================
# Session State Initialization
# ============================================================
# All analysis results and UI state are stored in session_state
# to persist across Streamlit reruns.
if "basic_results" not in st.session_state:
    st.session_state.basic_results = None
if "pro_results" not in st.session_state:
    st.session_state.pro_results = None
if "extracted_texts" not in st.session_state:
    st.session_state.extracted_texts = {}
if "job_type_detected" not in st.session_state:
    st.session_state.job_type_detected = None
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "pro_complete" not in st.session_state:
    st.session_state.pro_complete = False


# ============================================================
# Header
# ============================================================
st.markdown(
    """
    <div style="text-align: center; padding: 1rem 0 0.5rem;">
        <h1 style="
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(135deg, #8B5CF6 0%, #3B82F6 50%, #06B6D4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.3rem;
            letter-spacing: -0.02em;
        ">🔬 ResumeAI</h1>
        <p style="
            color: #94A3B8;
            font-size: 1.05rem;
            font-weight: 400;
            letter-spacing: 0.01em;
        ">Semantic Resume Screening — Beyond Keywords</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")


# ============================================================
# Sidebar — Input Controls
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding-bottom: 0.5rem;">
            <h2 style="
                font-size: 1.3rem;
                font-weight: 700;
                color: #F1F5F9;
                margin-bottom: 0.2rem;
            ">⚙️ Configuration</h2>
            <p style="color: #64748B; font-size: 0.85rem;">Set up your screening criteria</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ---- Job Description Input ----
    st.markdown("##### 📋 Job Description")
    jd_text = st.text_area(
        "Paste the full job description",
        height=200,
        placeholder="Paste the complete job description here...\n\nExample: We are looking for a Senior Data Scientist with 5+ years of experience in Python, Machine Learning, and SQL...",
        label_visibility="collapsed",
        key="jd_input",
    )

    st.markdown("")

    # ---- Job Type Input (Optional) ----
    st.markdown("##### 🏷️ Job Type *(optional)*")
    job_type_input = st.text_input(
        "e.g., Data Scientist, Backend Engineer",
        placeholder="Leave blank for AI auto-detection",
        label_visibility="collapsed",
        key="job_type_input",
    )
    if not job_type_input:
        st.caption("💡 *If left blank, Gemini will auto-detect the job type from the JD.*")

    st.markdown("")

    # ---- File Uploader ----
    st.markdown("##### 📎 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="file_uploader",
    )

    if uploaded_files:
        st.caption(f"📂 **{len(uploaded_files)}** file(s) uploaded")

    st.markdown("---")

    # ---- Analyze Button ----
    analyze_clicked = st.button(
        "🔍  Analyze Resumes",
        width="stretch",
        type="primary",
        key="analyze_btn",
    )

    st.markdown("")

    # ---- API Status Indicator ----
    client = init_client()
    if client:
        st.success("🟢 Gemini API Connected", icon="✅")
    else:
        st.warning(
            "🟡 API key not configured. "
            "BASIC mode works without it. "
            "PRO mode requires a valid API key in `.streamlit/secrets.toml`.",
            icon="⚠️",
        )


# ============================================================
# Main Content Area
# ============================================================

# Validation check
if analyze_clicked:
    if not jd_text.strip():
        st.error("❌ Please paste a Job Description in the sidebar.", icon="🚫")
        st.stop()
    if not uploaded_files:
        st.error("❌ Please upload at least one resume file.", icon="🚫")
        st.stop()


# ============================================================
# BASIC MODE — Analysis Pipeline
# ============================================================
if analyze_clicked and jd_text.strip() and uploaded_files:
    # Reset previous results
    st.session_state.basic_results = None
    st.session_state.pro_results = None
    st.session_state.pro_complete = False
    st.session_state.extracted_texts = {}

    # ---- Step 1: Job Type Detection ----
    with st.status("⚡ Running BASIC Analysis...", expanded=True) as status:

        # Determine job type
        if job_type_input.strip():
            st.session_state.job_type_detected = job_type_input.strip()
            st.write(f"🏷️ Job Type: **{st.session_state.job_type_detected}** *(user-provided)*")
        else:
            if client:
                st.write("🤖 Auto-detecting job type via Gemini...")
                st.session_state.job_type_detected = auto_detect_job_type(client, jd_text)
                st.write(f"🏷️ Job Type: **{st.session_state.job_type_detected}** *(AI-detected)*")
            else:
                st.session_state.job_type_detected = "General"
                st.write("🏷️ Job Type: **General** *(API not configured for auto-detection)*")

        # ---- Step 2: Extract Text from All Resumes ----
        st.write("📄 Extracting text from resumes...")
        progress_bar = st.progress(0)
        extraction_errors = []

        for i, file in enumerate(uploaded_files):
            extracted = extract_text(file)
            if extracted:
                st.session_state.extracted_texts[file.name] = extracted
            else:
                extraction_errors.append(file.name)
            progress_bar.progress((i + 1) / len(uploaded_files))

        if extraction_errors:
            st.warning(
                f"⚠️ Could not extract text from: {', '.join(extraction_errors)}"
            )

        if not st.session_state.extracted_texts:
            st.error("❌ No text could be extracted from any uploaded file.")
            status.update(label="❌ Analysis Failed", state="error")
            st.stop()

        st.write(
            f"✅ Successfully extracted text from "
            f"**{len(st.session_state.extracted_texts)}** resume(s)"
        )

        # ---- Step 3: Run BASIC Analysis on Each Resume ----
        st.write("🔍 Running keyword analysis...")
        basic_results = []
        progress_bar2 = st.progress(0)

        for i, (fname, rtext) in enumerate(st.session_state.extracted_texts.items()):
            result = analyze_resume_basic(jd_text, rtext, fname)
            basic_results.append(result)
            progress_bar2.progress((i + 1) / len(st.session_state.extracted_texts))

        # Sort by composite score descending
        basic_results.sort(key=lambda x: x["composite_score"], reverse=True)

        st.session_state.basic_results = basic_results
        st.session_state.analysis_complete = True

        status.update(label="✅ BASIC Analysis Complete!", state="complete")


# ============================================================
# Display BASIC Results
# ============================================================
if st.session_state.basic_results:
    results = st.session_state.basic_results

    # ---- Section Header ----
    st.markdown(
        """
        <div style="padding: 0.5rem 0;">
            <h2 style="
                font-size: 1.8rem;
                font-weight: 700;
                color: #F1F5F9;
                margin-bottom: 0.2rem;
            ">⚡ BASIC Analysis</h2>
            <p style="color: #64748B; font-size: 0.9rem;">
                TF-IDF keyword matching • Section detection • Contact extraction
                <span style="
                    display: inline-block;
                    background: rgba(16, 185, 129, 0.12);
                    color: #10B981;
                    padding: 2px 10px;
                    border-radius: 100px;
                    font-size: 0.78rem;
                    font-weight: 600;
                    margin-left: 8px;
                    border: 1px solid rgba(16, 185, 129, 0.25);
                ">ZERO TOKENS</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Job Type Badge ----
    if st.session_state.job_type_detected:
        st.markdown(
            f"""
            <div style="margin-bottom: 1rem;">
                <span style="
                    background: rgba(139, 92, 246, 0.12);
                    color: #A78BFA;
                    padding: 5px 14px;
                    border-radius: 100px;
                    font-size: 0.85rem;
                    font-weight: 600;
                    border: 1px solid rgba(139, 92, 246, 0.25);
                ">🏷️ {st.session_state.job_type_detected}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Summary Metrics Row ----
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📑 Resumes Analyzed", len(results))
    with col2:
        avg_score = sum(r["composite_score"] for r in results) / len(results)
        st.metric("📊 Avg. ATS Score", f"{avg_score:.1f}%")
    with col3:
        top_score = results[0]["composite_score"] if results else 0
        st.metric("🏆 Top Score", f"{top_score:.1f}%")
    with col4:
        avg_keywords = sum(r["keyword_match_count"] for r in results) / len(results)
        st.metric("🔑 Avg. Keywords", f"{avg_keywords:.0f}")

    st.markdown("")

    # ---- Leaderboard Table ----
    st.markdown("#### 🏅 Candidate Leaderboard")

    leaderboard_data = []
    for i, r in enumerate(results):
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
        leaderboard_data.append({
            "Rank": rank_emoji,
            "Candidate": r["file_name"],
            "ATS Score": f"{r['ats_score']}%",
            "Keywords Matched": f"{r['keyword_match_count']}/{r['keyword_total']}",
            "Section Score": f"{r['section_score']}%",
            "Composite Score": f"{r['composite_score']}%",
        })

    df = pd.DataFrame(leaderboard_data)
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "Rank": st.column_config.TextColumn("Rank", width="small"),
            "Candidate": st.column_config.TextColumn("Candidate", width="medium"),
            "Composite Score": st.column_config.TextColumn("Composite Score", width="small"),
        },
    )

    st.markdown("")

    # ---- Detailed Results per Candidate ----
    st.markdown("#### 📋 Detailed Breakdown")

    for i, r in enumerate(results):
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
        score_color = (
            "#10B981" if r["composite_score"] >= 70
            else "#F59E0B" if r["composite_score"] >= 40
            else "#EF4444"
        )

        with st.expander(
            f"{rank_emoji}  {r['file_name']}  —  Composite: {r['composite_score']}%",
            expanded=(i == 0),  # Auto-expand the top candidate
        ):
            # ---- Score Cards Row ----
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.metric("🎯 ATS Score", f"{r['ats_score']}%")
            with sc2:
                st.metric("🔑 Keyword Match", f"{r['keyword_match_count']}/{r['keyword_total']}")
            with sc3:
                st.metric("📑 Section Score", f"{r['section_score']}%")
            with sc4:
                st.metric("📝 Word Count", f"{r['word_count']}")

            st.markdown("")

            # ---- Keywords Section ----
            kw_col1, kw_col2 = st.columns(2)

            with kw_col1:
                st.markdown("**✅ Matched Keywords**")
                if r["matched_keywords"]:
                    # Render as green pills
                    pills_html = " ".join(
                        f'<span style="display:inline-block;background:rgba(16,185,129,0.12);color:#10B981;padding:3px 10px;border-radius:100px;font-size:0.8rem;font-weight:500;margin:2px 3px;border:1px solid rgba(16,185,129,0.25);">{kw}</span>'
                        for kw in r["matched_keywords"]
                    )
                    st.markdown(pills_html, unsafe_allow_html=True)
                else:
                    st.caption("No keyword matches found.")

            with kw_col2:
                st.markdown("**❌ Missing Keywords**")
                if r["missing_keywords"]:
                    # Render as red pills
                    pills_html = " ".join(
                        f'<span style="display:inline-block;background:rgba(239,68,68,0.12);color:#EF4444;padding:3px 10px;border-radius:100px;font-size:0.8rem;font-weight:500;margin:2px 3px;border:1px solid rgba(239,68,68,0.25);">{kw}</span>'
                        for kw in r["missing_keywords"]
                    )
                    st.markdown(pills_html, unsafe_allow_html=True)
                else:
                    st.caption("All key terms covered! 🎉")

            st.markdown("")

            # ---- Sections & Contact Info ----
            info_col1, info_col2 = st.columns(2)

            with info_col1:
                st.markdown("**📑 Resume Sections**")
                for section, found in r["sections"].items():
                    icon = "✅" if found else "❌"
                    st.markdown(f"&nbsp;&nbsp;{icon} {section}")

            with info_col2:
                st.markdown("**📬 Contact Information**")
                contact = r["contact_info"]
                if contact.get("email"):
                    st.markdown(f"&nbsp;&nbsp;📧 {contact['email']}")
                else:
                    st.markdown("&nbsp;&nbsp;📧 ❌ No email found")

                if contact.get("phone"):
                    st.markdown(f"&nbsp;&nbsp;📱 {contact['phone']}")
                else:
                    st.markdown("&nbsp;&nbsp;📱 ❌ No phone found")

                if contact.get("linkedin"):
                    st.markdown(f"&nbsp;&nbsp;🔗 {contact['linkedin']}")
                if contact.get("github"):
                    st.markdown(f"&nbsp;&nbsp;💻 {contact['github']}")

                st.markdown(f"&nbsp;&nbsp;📄 ~{r['estimated_pages']} page(s)")

    # ============================================================
    # PRO MODE — Unlock Button
    # ============================================================
    st.markdown("---")

    if not st.session_state.pro_complete:
        st.markdown(
            """
            <div style="text-align: center; padding: 2rem 0 1rem;">
                <h2 style="
                    font-size: 1.6rem;
                    font-weight: 700;
                    background: linear-gradient(135deg, #F59E0B 0%, #EF4444 50%, #8B5CF6 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    margin-bottom: 0.5rem;
                ">🧠 Ready for Deeper Insights?</h2>
                <p style="color: #94A3B8; font-size: 0.95rem; max-width: 600px; margin: 0 auto;">
                    PRO mode uses <strong>Gemini AI vector embeddings</strong> for semantic matching
                    and <strong>LLM contextual analysis</strong> to find skills traditional keyword
                    matching misses.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Center the PRO button
        _, pro_col, _ = st.columns([1, 2, 1])
        with pro_col:
            with st.container(key="pro-btn"):
                pro_clicked = st.button(
                    "🚀  Unlock PRO Analysis",
                    width="stretch",
                    key="pro_analysis_btn",
                )

        if pro_clicked:
            if not client:
                st.error(
                    "❌ **Gemini API key required for PRO mode.** "
                    "Add your key to `.streamlit/secrets.toml` and restart the app.",
                    icon="🔑",
                )
            else:
                # ---- Run PRO Analysis ----
                with st.status("🧠 Running PRO Analysis...", expanded=True) as pro_status:
                    st.write("🔗 Generating semantic embeddings with Gemini...")

                    pro_results = []
                    progress_pro = st.progress(0)
                    total = len(st.session_state.extracted_texts)

                    for i, (fname, rtext) in enumerate(
                        st.session_state.extracted_texts.items()
                    ):
                        st.write(f"🔬 Analyzing: {fname}")
                        result = analyze_resume_pro(
                            client,
                            jd_text,
                            rtext,
                            fname,
                            st.session_state.job_type_detected or "General",
                        )
                        pro_results.append(result)
                        progress_pro.progress((i + 1) / total)

                    # Sort by semantic score descending
                    pro_results.sort(
                        key=lambda x: x["semantic_score"], reverse=True
                    )

                    st.session_state.pro_results = pro_results
                    st.session_state.pro_complete = True

                    pro_status.update(
                        label="✅ PRO Analysis Complete!", state="complete"
                    )

                st.rerun()


    # ============================================================
    # Display PRO Results
    # ============================================================
    if st.session_state.pro_complete and st.session_state.pro_results:
        pro_results = st.session_state.pro_results

        st.markdown(
            """
            <div style="padding: 0.5rem 0;">
                <h2 style="
                    font-size: 1.8rem;
                    font-weight: 700;
                    background: linear-gradient(135deg, #F59E0B 0%, #EF4444 50%, #8B5CF6 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    margin-bottom: 0.2rem;
                ">🧠 PRO Analysis</h2>
                <p style="color: #64748B; font-size: 0.9rem;">
                    Semantic Vector Embeddings • LLM Contextual Extraction • AI Fit Assessment
                    <span style="
                        display: inline-block;
                        background: rgba(139, 92, 246, 0.15);
                        color: #A78BFA;
                        padding: 2px 10px;
                        border-radius: 100px;
                        font-size: 0.78rem;
                        font-weight: 600;
                        margin-left: 8px;
                        border: 1px solid rgba(139, 92, 246, 0.3);
                    ">GEMINI POWERED</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- PRO Summary Metrics ----
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            avg_semantic = sum(r["semantic_score"] for r in pro_results) / len(pro_results)
            st.metric("🧠 Avg. Semantic Score", f"{avg_semantic:.1f}%")
        with pc2:
            top_semantic = pro_results[0]["semantic_score"] if pro_results else 0
            st.metric("🏆 Top Semantic Score", f"{top_semantic:.1f}%")
        with pc3:
            # Compare BASIC vs PRO top scorer
            basic_top = st.session_state.basic_results[0]["composite_score"] if st.session_state.basic_results else 0
            delta = top_semantic - basic_top
            delta_label = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
            st.metric("📊 Score Delta (vs BASIC)", delta_label)

        st.markdown("")

        # ---- PRO Leaderboard ----
        st.markdown("#### 🏅 Semantic Leaderboard")

        pro_leaderboard = []
        for i, r in enumerate(pro_results):
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"

            # Find the corresponding BASIC score for comparison
            basic_match = next(
                (b for b in st.session_state.basic_results if b["file_name"] == r["file_name"]),
                None,
            )
            basic_score = basic_match["composite_score"] if basic_match else "N/A"

            pro_leaderboard.append({
                "Rank": rank_emoji,
                "Candidate": r["file_name"],
                "Semantic Score": f"{r['semantic_score']}%",
                "Basic Score": f"{basic_score}%" if isinstance(basic_score, (int, float)) else basic_score,
                "Fit Level": r["fit_level"],
            })

        pro_df = pd.DataFrame(pro_leaderboard)
        st.dataframe(pro_df, width="stretch", hide_index=True)

        st.markdown("")

        # ---- Detailed PRO Results per Candidate ----
        st.markdown("#### 🔬 AI Insights per Candidate")

        for i, r in enumerate(pro_results):
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"

            # Fit level color
            fit_colors = {
                "Strong Fit": ("#10B981", "rgba(16,185,129,0.12)", "rgba(16,185,129,0.25)"),
                "Moderate Fit": ("#F59E0B", "rgba(245,158,11,0.12)", "rgba(245,158,11,0.25)"),
                "Weak Fit": ("#EF4444", "rgba(239,68,68,0.12)", "rgba(239,68,68,0.25)"),
            }
            fc, fb, fbo = fit_colors.get(
                r["fit_level"], ("#94A3B8", "rgba(148,163,184,0.12)", "rgba(148,163,184,0.25)")
            )

            with st.expander(
                f"{rank_emoji}  {r['file_name']}  —  Semantic: {r['semantic_score']}%  |  {r['fit_level']}",
                expanded=(i == 0),
            ):
                # ---- Score + Fit Badge ----
                badge_col1, badge_col2 = st.columns([1, 3])
                with badge_col1:
                    st.metric("🧠 Semantic Score", f"{r['semantic_score']}%")
                with badge_col2:
                    st.markdown(
                        f"""
                        <div style="padding-top: 0.5rem;">
                            <span style="
                                display: inline-block;
                                background: {fb};
                                color: {fc};
                                padding: 6px 16px;
                                border-radius: 100px;
                                font-size: 0.95rem;
                                font-weight: 600;
                                border: 1px solid {fbo};
                            ">{r['fit_level']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("")

                # ---- Fit Assessment ----
                st.markdown("**🎯 AI Assessment**")
                st.markdown(
                    f"""
                    <div style="
                        background: rgba(255,255,255,0.03);
                        border-left: 3px solid #8B5CF6;
                        padding: 0.8rem 1rem;
                        border-radius: 0 8px 8px 0;
                        color: #CBD5E1;
                        font-size: 0.92rem;
                        line-height: 1.6;
                    ">{r['fit_assessment']}</div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("")

                # ---- Skills ----
                sk_col1, sk_col2 = st.columns(2)

                with sk_col1:
                    st.markdown("**✅ Contextually Matched Skills**")
                    if r["matched_skills"]:
                        pills = " ".join(
                            f'<span style="display:inline-block;background:rgba(16,185,129,0.12);color:#10B981;padding:4px 12px;border-radius:100px;font-size:0.82rem;font-weight:500;margin:3px;border:1px solid rgba(16,185,129,0.25);">{s}</span>'
                            for s in r["matched_skills"]
                        )
                        st.markdown(pills, unsafe_allow_html=True)
                    else:
                        st.caption("No contextual matches identified.")

                with sk_col2:
                    st.markdown("**❌ Critical Missing Skills**")
                    if r["missing_skills"]:
                        pills = " ".join(
                            f'<span style="display:inline-block;background:rgba(239,68,68,0.12);color:#EF4444;padding:4px 12px;border-radius:100px;font-size:0.82rem;font-weight:500;margin:3px;border:1px solid rgba(239,68,68,0.25);">{s}</span>'
                            for s in r["missing_skills"]
                        )
                        st.markdown(pills, unsafe_allow_html=True)
                    else:
                        st.caption("No critical gaps identified! 🎉")


# ============================================================
# Empty State (No analysis run yet)
# ============================================================
if not st.session_state.basic_results and not analyze_clicked:
    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 4rem 2rem;
            color: #64748B;
        ">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🔬</div>
            <h3 style="
                color: #94A3B8;
                font-weight: 600;
                font-size: 1.4rem;
                margin-bottom: 0.8rem;
            ">No Analysis Yet</h3>
            <p style="max-width: 500px; margin: 0 auto; line-height: 1.7; font-size: 0.95rem;">
                Paste a <strong style="color: #A78BFA;">Job Description</strong> in the sidebar,
                upload your <strong style="color: #A78BFA;">resumes</strong> (PDF/DOCX),
                and click <strong style="color: #A78BFA;">Analyze</strong> to get started.
            </p>
            <div style="
                margin-top: 2rem;
                padding: 1.2rem 1.5rem;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 12px;
                display: inline-block;
                text-align: left;
                max-width: 420px;
            ">
                <p style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 0.4rem; font-weight: 600;">How it works:</p>
                <p style="font-size: 0.85rem; margin: 0.2rem 0;">⚡ <strong>BASIC</strong> — Instant keyword analysis (free, no API needed)</p>
                <p style="font-size: 0.85rem; margin: 0.2rem 0;">🧠 <strong>PRO</strong> — Gemini semantic AI analysis (optional, on-demand)</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; padding: 0.5rem 0; color: #475569; font-size: 0.8rem;">
        Built with ❤️ using Streamlit & Google Gemini AI  •  Capstone Project 2026
    </div>
    """,
    unsafe_allow_html=True,
)
