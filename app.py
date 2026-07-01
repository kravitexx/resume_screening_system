# ============================================================
# app.py — Semantic Resume Screening System
# ============================================================
# A two-tier resume screening application with Compare mode:
#   BASIC Mode: TF-IDF keyword matching (zero API tokens)
#   PRO Mode:   Gemini embeddings + LLM analysis (on-demand)
# ============================================================

import streamlit as st
import pandas as pd
import os

# Import our utility modules
from utils.text_extraction import extract_text
from utils.basic_analyzer import analyze_resume_basic
from utils.gemini_client import init_client, auto_detect_job_type
from utils.pro_analyzer import analyze_resume_pro
from utils.compare_analyzer import compare_candidates_pro
from utils.export_utils import generate_markdown_report, generate_docx_report, generate_pdf_report


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
# Load Custom CSS (Dark Glassmorphism)
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
if "job_type_detected" not in st.session_state:
    st.session_state.job_type_detected = None

# Single mode states
if "single_basic_results" not in st.session_state:
    st.session_state.single_basic_results = None
if "single_pro_results" not in st.session_state:
    st.session_state.single_pro_results = None
if "single_pro_complete" not in st.session_state:
    st.session_state.single_pro_complete = False
if "single_extracted_text" not in st.session_state:
    st.session_state.single_extracted_text = {}

# Compare mode states
if "compare_basic_results" not in st.session_state:
    st.session_state.compare_basic_results = None
if "compare_pro_summary" not in st.session_state:
    st.session_state.compare_pro_summary = None
if "compare_pro_complete" not in st.session_state:
    st.session_state.compare_pro_complete = False
if "compare_extracted_texts" not in st.session_state:
    st.session_state.compare_extracted_texts = {}


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
        ">Semantic Resume Screening & Comparison</p>
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
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("##### 📋 Job Description")
    jd_text = st.text_area(
        "Paste the full job description",
        height=200,
        placeholder="Paste the complete job description here...",
        label_visibility="collapsed",
        key="jd_input",
    )

    st.markdown("")

    st.markdown("##### 🏷️ Job Type *(optional)*")
    job_type_input = st.text_input(
        "e.g., Data Scientist, Backend Engineer",
        placeholder="Leave blank for AI auto-detection",
        label_visibility="collapsed",
        key="job_type_input",
    )

    st.markdown("---")

    client = init_client()
    if client:
        st.success("🟢 Gemini API Connected", icon="✅")
    else:
        st.warning(
            "🟡 API key not configured. BASIC mode works without it.",
            icon="⚠️",
        )


def detect_job_type_if_needed():
    if job_type_input.strip():
        st.session_state.job_type_detected = job_type_input.strip()
    else:
        if client:
            st.session_state.job_type_detected = auto_detect_job_type(client, jd_text)
        else:
            st.session_state.job_type_detected = "General"


# ============================================================
# TABS: Single Screening | Compare Resumes
# ============================================================
tab1, tab2 = st.tabs(["🎯 Single Screening", "⚖️ Compare Resumes"])


# ============================================================
# TAB 1: Single Screening
# ============================================================
with tab1:
    st.markdown("### 📎 Upload Candidate Resume")
    single_file = st.file_uploader(
        "Upload a single PDF or DOCX file",
        type=["pdf", "docx"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key="single_uploader",
    )
    
    if single_file:
        st.caption(f"📂 **{single_file.name}** uploaded")

    analyze_single_btn = st.button("🔍 Analyze Resume", type="primary", key="analyze_single")

    if analyze_single_btn:
        if not jd_text.strip():
            st.error("❌ Please paste a Job Description in the sidebar.")
        elif not single_file:
            st.error("❌ Please upload a resume.")
        else:
            st.session_state.single_basic_results = None
            st.session_state.single_pro_results = None
            st.session_state.single_pro_complete = False
            st.session_state.single_extracted_text = {}

            with st.status("⚡ Running BASIC Analysis...", expanded=True) as status:
                detect_job_type_if_needed()
                
                extracted = extract_text(single_file)
                if not extracted:
                    status.update(label="❌ Analysis Failed: No text extracted", state="error")
                    st.stop()
                    
                st.session_state.single_extracted_text[single_file.name] = extracted
                result = analyze_resume_basic(jd_text, extracted, single_file.name)
                
                st.session_state.single_basic_results = [result]
                status.update(label="✅ BASIC Analysis Complete!", state="complete")

    if st.session_state.single_basic_results:
        results = st.session_state.single_basic_results
        r = results[0]

        # BASIC UI Flourishes
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

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("🎯 ATS Score", f"{r['ats_score']}%")
        sc2.metric("🔑 Keyword Match", f"{r['keyword_match_count']}/{r['keyword_total']}")
        sc3.metric("📑 Section Score", f"{r['section_score']}%")
        sc4.metric("📝 Word Count", f"{r['word_count']}")

        with st.expander(f"🏅 Detailed Breakdown for {r['file_name']}", expanded=True):
            kw_col1, kw_col2 = st.columns(2)
            with kw_col1:
                st.markdown("**✅ Matched Keywords**")
                if r["matched_keywords"]:
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
                    pills_html = " ".join(
                        f'<span style="display:inline-block;background:rgba(239,68,68,0.12);color:#EF4444;padding:3px 10px;border-radius:100px;font-size:0.8rem;font-weight:500;margin:2px 3px;border:1px solid rgba(239,68,68,0.25);">{kw}</span>'
                        for kw in r["missing_keywords"]
                    )
                    st.markdown(pills_html, unsafe_allow_html=True)
                else:
                    st.caption("All key terms covered! 🎉")
                    
            st.markdown("---")
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.markdown("**📑 Resume Sections**")
                for section, found in r["sections"].items():
                    icon = "✅" if found else "❌"
                    st.markdown(f"&nbsp;&nbsp;{icon} {section}")
            with info_col2:
                st.markdown("**📬 Contact Information**")
                contact = r["contact_info"]
                st.markdown(f"&nbsp;&nbsp;📧 {contact.get('email', '❌ No email')}")
                st.markdown(f"&nbsp;&nbsp;📱 {contact.get('phone', '❌ No phone')}")

        st.markdown("---")

        # PRO Single Mode UI
        if not st.session_state.single_pro_complete:
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
            _, pro_col, _ = st.columns([1, 2, 1])
            with pro_col:
                if st.button("🚀 Unlock PRO Analysis for Candidate", width="stretch", key="pro_single_btn"):
                    if not client:
                        st.error("❌ Gemini API key required in `.streamlit/secrets.toml`")
                    else:
                        with st.status("🧠 Running PRO Analysis...", expanded=True) as pro_status:
                            fname = single_file.name if single_file else list(st.session_state.single_extracted_text.keys())[0]
                            rtext = st.session_state.single_extracted_text[fname]
                            res = analyze_resume_pro(client, jd_text, rtext, fname, st.session_state.job_type_detected)
                            
                            st.session_state.single_pro_results = [res]
                            st.session_state.single_pro_complete = True
                            pro_status.update(label="✅ PRO Analysis Complete!", state="complete")
                        st.rerun()

        if st.session_state.single_pro_complete and st.session_state.single_pro_results:
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
            pr = st.session_state.single_pro_results[0]
            
            fit_colors = {
                "Strong Fit": ("#10B981", "rgba(16,185,129,0.12)", "rgba(16,185,129,0.25)"),
                "Moderate Fit": ("#F59E0B", "rgba(245,158,11,0.12)", "rgba(245,158,11,0.25)"),
                "Weak Fit": ("#EF4444", "rgba(239,68,68,0.12)", "rgba(239,68,68,0.25)"),
            }
            fc, fb, fbo = fit_colors.get(pr["fit_level"], ("#94A3B8", "rgba(148,163,184,0.12)", "rgba(148,163,184,0.25)"))

            p1, p2 = st.columns([1, 3])
            p1.metric("🧠 Semantic Score", f"{pr['semantic_score']}%")
            with p2:
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
                        ">{pr['fit_level']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            st.markdown("**🎯 AI Assessment**")
            import markdown
            parsed_assessment = markdown.markdown(pr['fit_assessment'])
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
                ">{parsed_assessment}</div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("")
            
            sk1, sk2 = st.columns(2)
            with sk1:
                st.markdown("**✅ Contextually Matched Skills**")
                if pr["matched_skills"]:
                    pills = " ".join(
                        f'<span style="display:inline-block;background:rgba(16,185,129,0.12);color:#10B981;padding:4px 12px;border-radius:100px;font-size:0.82rem;font-weight:500;margin:3px;border:1px solid rgba(16,185,129,0.25);">{s}</span>'
                        for s in pr["matched_skills"]
                    )
                    st.markdown(pills, unsafe_allow_html=True)
                else:
                    st.caption("None identified.")
            with sk2:
                st.markdown("**❌ Critical Missing Skills**")
                if pr["missing_skills"]:
                    pills = " ".join(
                        f'<span style="display:inline-block;background:rgba(239,68,68,0.12);color:#EF4444;padding:4px 12px;border-radius:100px;font-size:0.82rem;font-weight:500;margin:3px;border:1px solid rgba(239,68,68,0.25);">{s}</span>'
                        for s in pr["missing_skills"]
                    )
                    st.markdown(pills, unsafe_allow_html=True)
                else:
                    st.caption("No critical gaps identified! 🎉")
                    
        # Export Buttons
        st.markdown("### 💾 Export Single Screening Report")
        col_md, col_docx, col_pdf = st.columns(3)
        export_data = st.session_state.single_pro_results if st.session_state.single_pro_complete else st.session_state.single_basic_results
        is_pro = st.session_state.single_pro_complete
        
        md_text = generate_markdown_report(export_data, is_pro)
        col_md.download_button("📝 Download Markdown", data=md_text, file_name="resume_report.md", mime="text/markdown", width="stretch", key="single_md")
        
        docx_file = generate_docx_report(export_data, is_pro)
        col_docx.download_button("📄 Download DOCX", data=docx_file, file_name="resume_report.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", width="stretch", key="single_docx")
        
        pdf_bytes = generate_pdf_report(export_data, is_pro)
        col_pdf.download_button("📕 Download PDF", data=pdf_bytes, file_name="resume_report.pdf", mime="application/pdf", width="stretch", key="single_pdf")

    else:
        st.info("Upload a single resume and click Analyze to view results.")


# ============================================================
# TAB 2: Compare Resumes
# ============================================================
with tab2:
    st.markdown("### 📎 Upload Multiple Resumes")
    compare_files = st.file_uploader(
        "Upload 2 to 4 PDF or DOCX files for comparison",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="compare_uploader",
    )
    
    if compare_files:
        st.caption(f"📂 **{len(compare_files)}** file(s) uploaded")

    analyze_compare_btn = st.button("⚖️ Compare Resumes", type="primary", key="analyze_compare")

    if analyze_compare_btn:
        if not jd_text.strip():
            st.error("❌ Please paste a Job Description in the sidebar.")
        elif not compare_files or len(compare_files) < 2:
            st.error("❌ Please upload at least 2 resumes to compare.")
        else:
            st.session_state.compare_basic_results = None
            st.session_state.compare_pro_summary = None
            st.session_state.compare_pro_complete = False
            st.session_state.compare_extracted_texts = {}

            with st.status("⚡ Running BASIC Comparison Pipeline...", expanded=True) as status:
                detect_job_type_if_needed()
                
                # Limit to 4 files internally
                files_to_process = compare_files[:4]
                if len(compare_files) > 4:
                    st.warning("⚠️ Only the first 4 resumes will be used for optimal comparison.")

                for file in files_to_process:
                    extracted = extract_text(file)
                    if extracted:
                        st.session_state.compare_extracted_texts[file.name] = extracted

                if not st.session_state.compare_extracted_texts:
                    status.update(label="❌ Analysis Failed: No text extracted", state="error")
                    st.stop()

                compare_results = []
                for fname, rtext in st.session_state.compare_extracted_texts.items():
                    result = analyze_resume_basic(jd_text, rtext, fname)
                    compare_results.append(result)

                compare_results.sort(key=lambda x: x["composite_score"], reverse=True)
                st.session_state.compare_basic_results = compare_results
                status.update(label="✅ BASIC Comparison Complete!", state="complete")


    if st.session_state.compare_basic_results:
        results = st.session_state.compare_basic_results
        
        st.markdown(
            """
            <div style="padding: 0.5rem 0;">
                <h2 style="
                    font-size: 1.8rem;
                    font-weight: 700;
                    color: #F1F5F9;
                    margin-bottom: 0.2rem;
                ">⚖️ Candidate Comparison Matrix</h2>
                <p style="color: #64748B; font-size: 0.9rem;">
                    Side-by-side BASIC metrics
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # DataFrame Leaderboard style for Comparison
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
        
        # Display Side-by-Side Detail Expanders
        for i, res in enumerate(results):
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
            with st.expander(f"{rank_emoji}  {res['file_name']}  —  Composite: {res['composite_score']}%", expanded=(i==0)):
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("🎯 ATS Score", f"{res['ats_score']}%")
                sc2.metric("🔑 Keyword Match", f"{res['keyword_match_count']}/{res['keyword_total']}")
                sc3.metric("📑 Section Score", f"{res['section_score']}%")
                sc4.metric("📝 Word Count", f"{res['word_count']}")
                
                kw_col1, kw_col2 = st.columns(2)
                with kw_col1:
                    st.markdown("**✅ Matched Keywords**")
                    if res["matched_keywords"]:
                        pills = " ".join(
                            f'<span style="display:inline-block;background:rgba(16,185,129,0.12);color:#10B981;padding:3px 10px;border-radius:100px;font-size:0.8rem;font-weight:500;margin:2px 3px;border:1px solid rgba(16,185,129,0.25);">{kw}</span>'
                            for kw in res["matched_keywords"]
                        )
                        st.markdown(pills, unsafe_allow_html=True)
                    else:
                        st.caption("No keyword matches found.")
                with kw_col2:
                    st.markdown("**❌ Missing Keywords**")
                    if res["missing_keywords"]:
                        pills = " ".join(
                            f'<span style="display:inline-block;background:rgba(239,68,68,0.12);color:#EF4444;padding:3px 10px;border-radius:100px;font-size:0.8rem;font-weight:500;margin:2px 3px;border:1px solid rgba(239,68,68,0.25);">{kw}</span>'
                            for kw in res["missing_keywords"]
                        )
                        st.markdown(pills, unsafe_allow_html=True)
                    else:
                        st.caption("All key terms covered! 🎉")

        st.markdown("---")
        
        # PRO Comparison
        if not st.session_state.compare_pro_complete:
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
                    ">🧠 AI Deep Comparison</h2>
                    <p style="color: #94A3B8; font-size: 0.95rem; max-width: 600px; margin: 0 auto;">
                        Let Gemini analyze these candidates side-by-side to determine who is truly the best fit and why.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            _, pro_col, _ = st.columns([1, 2, 1])
            with pro_col:
                if st.button("🚀 Run PRO Compare", width="stretch", key="pro_compare_btn"):
                    if not client:
                        st.error("❌ Gemini API key required in `.streamlit/secrets.toml`")
                    else:
                        with st.status("🧠 Running AI Comparison...", expanded=True):
                            top_4_filenames = [r["file_name"] for r in results]
                            compare_texts = {fname: st.session_state.compare_extracted_texts[fname] for fname in top_4_filenames}
                            result_md = compare_candidates_pro(
                                client, jd_text, compare_texts, st.session_state.job_type_detected
                            )
                            st.session_state.compare_pro_summary = result_md
                            st.session_state.compare_pro_complete = True
                        st.rerun()
        
        if st.session_state.compare_pro_complete:
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
                    ">🧠 AI Deep Comparison Result</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )
            import markdown
            parsed_summary = markdown.markdown(st.session_state.compare_pro_summary)
            
            st.markdown(
                f"""
                <div style="
                    background: rgba(255,255,255,0.03);
                    border-left: 3px solid #8B5CF6;
                    padding: 1.5rem;
                    border-radius: 0 8px 8px 0;
                    color: #E2E8F0;
                    font-size: 0.95rem;
                    line-height: 1.7;
                ">{parsed_summary}</div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("")
            
            # Export Buttons for Compare
            st.markdown("### 💾 Export Comparison Report")
            ccol_md, ccol_docx, ccol_pdf = st.columns(3)
            
            export_data_compare = results
            is_pro_compare = False
            
            md_text_c = generate_markdown_report(export_data_compare, is_pro_compare, st.session_state.compare_pro_summary)
            ccol_md.download_button("📝 Download Compare (MD)", data=md_text_c, file_name="resume_compare.md", mime="text/markdown", width="stretch", key="dl_md_c")
            
            docx_file_c = generate_docx_report(export_data_compare, is_pro_compare, st.session_state.compare_pro_summary)
            ccol_docx.download_button("📄 Download Compare (DOCX)", data=docx_file_c, file_name="resume_compare.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", width="stretch", key="dl_docx_c")
            
            pdf_bytes_c = generate_pdf_report(export_data_compare, is_pro_compare, st.session_state.compare_pro_summary)
            ccol_pdf.download_button("📕 Download Compare (PDF)", data=pdf_bytes_c, file_name="resume_compare.pdf", mime="application/pdf", width="stretch", key="dl_pdf_c")

    else:
        st.info("Upload 2-4 resumes and click Compare Resumes to see side-by-side analysis.")
