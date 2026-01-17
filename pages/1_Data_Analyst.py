
import streamlit as st

st.set_page_config(
    page_title="Data Analyst Tools",
    page_icon="📊",
    layout="wide"
)

# Dark theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }

    .page-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
    }

    .project-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }

    .project-card:hover {
        transform: translateY(-5px);
        border-color: rgba(102, 126, 234, 0.6);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }

    .project-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
    }

    .project-description {
        color: #b0b0b0;
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    .project-features {
        list-style: none;
        padding: 0;
        margin: 1rem 0;
    }

    .project-features li {
        color: #c0c0c0;
        padding: 0.4rem 0;
        padding-left: 1.5rem;
        position: relative;
    }

    .project-features li:before {
        content: "▹";
        position: absolute;
        left: 0;
        color: #667eea;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("### 📊 Data Analyst Tools")

    project = st.selectbox(
        "Select Tool",
        [
            "Overview",
            "📝 Resume Analyzer",
            "⚜️ Universal Analytics",
            "📋 CSV Q&A Analyst",
            "📑 PDF Document Analyzer"
        ]
    )

    st.markdown("---")

    if st.button("⬅️ Back to Portfolio"):
        st.switch_page("Portfolio_Dashboard.py")

# Main content
st.markdown('<h1 class="page-header"><span style="-webkit-text-fill-color: initial">📊</span> Data Analyst Tools</h1>', unsafe_allow_html=True)

if project == "Overview":
    st.markdown("""
    <div style="text-align: center; color: #a0a0a0; margin: 2rem 0;">
        <p style="font-size: 1.2rem;">
            Professional data analysis toolkit with 4 powerful applications
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Project 1: Resume Analyzer
    st.markdown("""
    <div class="project-card">
        <div class="project-title">📝 1. AI Resume Analyzer</div>
        <div class="project-description">
            Advanced ATS compatibility scoring with comprehensive keyword analysis and improvement recommendations
        </div>
        <ul class="project-features">
            <li>ATS compatibility scoring with detailed feedback</li>
            <li>Multi-category keyword analysis (Technical, Action Verbs, Soft Skills)</li>
            <li>Section detection and completeness check</li>
            <li>Experience level assessment</li>
            <li>Readability metrics and quantifiable achievements</li>
            <li>Interactive visualizations and comparison charts</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if st.button("📝 Launch Resume Analyzer", key="resume"):
        st.switch_page("pages/2_Resume_Analyzer.py")

    # Project 2: Universal Analytics
    st.markdown("""
    <div class="project-card">
        <div class="project-title">⚜️ 2. Universal Data Analytics</div>
        <div class="project-description">
            Automatically analyze ANY CSV file with comprehensive statistical analysis and professional reporting
        </div>
        <ul class="project-features">
            <li>Automatic column detection and smart data classification</li>
            <li>Comprehensive numerical and categorical analysis</li>
            <li>Correlation detection and relationship mapping</li>
            <li>Multiple visualizations (distributions, box plots, heatmaps, scatter plots)</li>
            <li>Automated insights generation</li>
            <li>Professional Word report export with publication-quality charts</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⚜️ Launch Universal Analytics", key="universal"):
        st.switch_page("pages/3_Universal_Analytics.py")

    # Project 3: CSV Q&A Analyst
    st.markdown("""
    <div class="project-card">
        <div class="project-title">📋 3. CSV Q&A Analyst</div>
        <div class="project-description">
            Ask questions about your data in natural language and get instant answers with visualizations
        </div>
        <ul class="project-features">
            <li>Natural language query processing</li>
            <li>Support for count, average, sum, max/min operations</li>
            <li>Automatic grouping and aggregation</li>
            <li>Correlation analysis and trend detection</li>
            <li>Auto-generated charts based on query type</li>
            <li>Multi-file support with data comparison</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if st.button("📋 Launch CSV Q&A Analyst", key="csv"):
        st.switch_page("pages/4_CSV_Analyzer.py")

    # Project 4: PDF Analyzer
    st.markdown("""
    <div class="project-card">
        <div class="project-title">📑 4. PDF Document Analyzer</div>
        <div class="project-description">
            Multi-algorithm document summarization with sentiment analysis and advanced NLP metrics
        </div>
        <ul class="project-features">
            <li>Three summarization algorithms (TextRank, LSA, Luhn)</li>
            <li>RAKE and TF-IDF keyword extraction</li>
            <li>Sentiment analysis with polarity and subjectivity scoring</li>
            <li>Readability metrics (Flesch Reading Ease)</li>
            <li>Entity detection and word frequency analysis</li>
            <li>Word cloud visualization and comprehensive reporting</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if st.button("📑 Launch PDF Analyzer", key="pdf"):
        st.switch_page("pages/5_PDF_Analyzer.py")

# Show selected project
elif project == "📝 Resume Analyzer":
    st.info("🔄 Redirecting to Resume Analyzer...")
    st.switch_page("pages/2_Resume_Analyzer.py")

elif project == "⚜️ Universal Analytics":
    st.info("🔄 Redirecting to Universal Analytics...")
    st.switch_page("pages/3_Universal_Analytics.py")

elif project == "📋 CSV Q&A Analyst":
    st.info("🔄 Redirecting to CSV Q&A Analyst...")
    st.switch_page("pages/4_CSV_Analyzer.py")

elif project == "📑 PDF Document Analyzer":
    st.info("🔄 Redirecting to PDF Analyzer...")
    st.switch_page("pages/5_PDF_Analyzer.py")
