
import streamlit as st

st.set_page_config(
    page_title="Data Analyst Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    .main-header {
        text-align: center;
        padding: 3rem 0 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4rem;
        font-weight: 800;
        margin-bottom: 1rem;
        animation: fadeInDown 1s ease-in;
    }

    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .subtitle {
        text-align: center;
        color: #a0a0a0;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }

    .tagline {
        text-align: center;
        color: #808080;
        font-size: 1.1rem;
        margin-bottom: 3rem;
    }

    .metrics-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 2rem 0 4rem 0;
        flex-wrap: wrap;
    }

    .metric-box {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1.5rem 2.5rem;
        text-align: center;
        min-width: 150px;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        color: #a0a0a0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .domain-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1rem;
        transition: all 0.3s ease;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    .domain-card:hover {
        transform: translateY(-10px);
        border-color: rgba(102, 126, 234, 0.6);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }

    .domain-icon {
        font-size: 4rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .domain-title {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1rem;
    }

    .domain-description {
        color: #b0b0b0;
        font-size: 1.1rem;
        line-height: 1.8;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .domain-features {
        list-style: none;
        padding: 0;
        margin: 1.5rem 0;
    }

    .domain-features li {
        color: #c0c0c0;
        padding: 0.5rem 0;
        padding-left: 1.5rem;
        position: relative;
    }

    .domain-features li:before {
        content: "✦";
        position: absolute;
        left: 0;
        color: #667eea;
    }

    .footer {
        text-align: center;
        color: #808080;
        padding: 3rem 0 2rem 0;
        margin-top: 4rem;
        border-top: 1px solid rgba(102, 126, 234, 0.2);
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header"><span style="-webkit-text-fill-color: initial">🃏</span>The Polymath Collection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">A Curated Portfolio Demonstrating the Synthesis of Diverse Expertise</p>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Bridging Technical, Creative, and Strategic Disciplines to Deliver Comprehensive and Holistic Solutions</p>', unsafe_allow_html=True)

    # Metrics row
    st.markdown("""
    <div class="metrics-container">
        <div class="metric-box">
            <div class="metric-value">4</div>
            <div class="metric-label">Productive Tools</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">∞</div>
            <div class="metric-label">Insights</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">1</div>
            <div class="metric-label">Categories</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">⚡</div>
            <div class="metric-label">Fast Analysis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Single domain card for Data Analyst
    st.markdown("""
    <div class="domain-card">
        <div class="domain-icon">📊</div>
        <div class="domain-title">Data Analyst Projects</div>
        <div class="domain-description">
            Professional data analysis toolkit with 4 powerful applications for comprehensive insights
        </div>
        <ul class="domain-features">
            <li><strong>Resume Analyzer:</strong> AI-powered ATS scoring and keyword analysis</li>
            <li><strong>Universal Analytics:</strong> Automated insights for any CSV dataset</li>
            <li><strong>CSV Q&A Analyst:</strong> Natural language queries on your data</li>
            <li><strong>PDF Analyzer:</strong> Document summarization and sentiment analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if st.button("📊 Explore Data Analyst Tools", key="da", use_container_width=True):
        st.switch_page("pages/1_Data_Analyst.py")

    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>Transforming Data into Actionable Insights</strong></p>
        <p>Built with ❤️ using Streamlit | Python | Advanced Analytics</p>
        <p>🔒 Secure • 🚀 Fast • 📊 Accurate • 🎯 Production-Ready</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
