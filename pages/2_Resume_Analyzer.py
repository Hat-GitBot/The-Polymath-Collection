
import streamlit as st
import fitz  # PyMuPDF
import re
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import pandas as pd
import numpy as np
from datetime import datetime
import textstat

# Configure page
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #1e1e1e !important;
        padding: 25px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
        border: 1px solid #333 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2.5rem !important;
        font-weight: bold !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        opacity: 0.9;
    }
    div[data-testid="stMetricDelta"] {
        color: #4ade80 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    /* Statistics section styling */
    .stats-header {
        color: #ffffff !important;
        font-weight: bold !important;
        font-size: 1.2rem !important;
        margin-bottom: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF with page-by-page breakdown"""
    try:
        # Read the file content first
        file_content = pdf_file.read()
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
        text = ""
        page_texts = []
        page_count = pdf_document.page_count

        for page_num in range(page_count):
            page = pdf_document[page_num]
            page_text = page.get_text()
            text += page_text + "\n"
            page_texts.append(page_text)

        pdf_document.close()
        return text, page_texts, page_count
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return "", [], 0

def detect_sections(text):
    """Detect common resume sections using pattern matching"""
    sections = {
        'contact': False,
        'summary': False,
        'experience': False,
        'education': False,
        'skills': False,
        'projects': False,
        'certifications': False,
        'awards': False
    }

    text_lower = text.lower()

    # Contact patterns
    if re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text) or re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text):
        sections['contact'] = True

    # Section headers
    if re.search(r'\b(summary|profile|objective|about)\b', text_lower):
        sections['summary'] = True
    if re.search(r'\b(experience|employment|work history)\b', text_lower):
        sections['experience'] = True
    if re.search(r'\b(education|academic|degree)\b', text_lower):
        sections['education'] = True
    if re.search(r'\b(skills|technical skills|competencies)\b', text_lower):
        sections['skills'] = True
    if re.search(r'\b(projects|portfolio)\b', text_lower):
        sections['projects'] = True
    if re.search(r'\b(certifications?|licenses?)\b', text_lower):
        sections['certifications'] = True
    if re.search(r'\b(awards?|honors?|achievements?)\b', text_lower):
        sections['awards'] = True

    return sections

def calculate_ats_score(text, sections):
    """Advanced ATS scoring based on multiple factors"""
    score = 0
    max_score = 100
    feedback = []

    # Text length check (20 points)
    text_length = len(text.strip())
    if text_length > 1000:
        score += 20
        feedback.append("✅ Sufficient text content")
    elif text_length > 500:
        score += 10
        feedback.append("⚠️ Moderate text content - consider adding more details")
    else:
        feedback.append("❌ Insufficient text - may be image-heavy or poorly formatted")

    # Section completeness (40 points)
    essential_sections = ['contact', 'experience', 'education', 'skills']
    sections_present = sum([sections[s] for s in essential_sections])
    section_score = (sections_present / len(essential_sections)) * 40
    score += section_score

    if sections_present == len(essential_sections):
        feedback.append("✅ All essential sections present")
    else:
        missing = [s.title() for s in essential_sections if not sections[s]]
        feedback.append(f"⚠️ Missing sections: {', '.join(missing)}")

    # Formatting checks (20 points)
    has_bullets = bool(re.search(r'[•●○▪▫■□]', text))
    has_dates = bool(re.search(r'\b(19|20)\d{2}\b', text))
    has_email = bool(re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text))
    has_phone = bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))

    format_checks = sum([has_bullets, has_dates, has_email, has_phone])
    score += (format_checks / 4) * 20

    if has_bullets:
        feedback.append("✅ Uses bullet points effectively")
    if has_email and has_phone:
        feedback.append("✅ Contact information present")

    # Readability (20 points)
    reading_ease = textstat.flesch_reading_ease(text[:1000])  # Sample first 1000 chars
    if reading_ease > 60:
        score += 20
        feedback.append("✅ Good readability score")
    elif reading_ease > 40:
        score += 10
        feedback.append("⚠️ Moderate readability - consider simplifying language")
    else:
        feedback.append("⚠️ Low readability - use simpler language")

    return min(score, max_score), feedback

def extract_advanced_keywords(text):
    """Extract keywords by category with weighted importance"""

    keyword_categories = {
        'Technical Skills': {
            'keywords': ['python', 'java', 'javascript', 'c++', 'sql', 'html', 'css', 'react', 'angular',
                        'node.js', 'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'tensorflow',
                        'pytorch', 'machine learning', 'deep learning', 'ai', 'data science',
                        'tableau', 'power bi', 'excel', 'r programming', 'mongodb', 'postgresql'],
            'weight': 1.5
        },
        'Action Verbs': {
            'keywords': ['managed', 'developed', 'implemented', 'created', 'designed', 'built',
                        'improved', 'increased', 'decreased', 'optimized', 'led', 'coordinated',
                        'supervised', 'analyzed', 'evaluated', 'trained', 'mentored', 'established',
                        'launched', 'achieved', 'delivered', 'executed', 'spearheaded'],
            'weight': 1.2
        },
        'Soft Skills': {
            'keywords': ['leadership', 'communication', 'teamwork', 'problem-solving', 'analytical',
                        'critical thinking', 'collaboration', 'adaptability', 'creativity', 'innovation',
                        'time management', 'organization', 'presentation', 'negotiation'],
            'weight': 1.0
        },
        'Business': {
            'keywords': ['strategy', 'revenue', 'growth', 'roi', 'kpi', 'metrics', 'budget',
                        'stakeholder', 'client', 'customer', 'market', 'sales', 'marketing',
                        'operations', 'project management', 'agile', 'scrum', 'product'],
            'weight': 1.3
        },
        'Certifications': {
            'keywords': ['certified', 'certification', 'pmp', 'aws certified', 'cisco', 'comptia',
                        'cpa', 'cfa', 'six sigma', 'scrum master', 'professional'],
            'weight': 1.4
        }
    }

    text_lower = text.lower()
    category_results = {}
    all_keywords = {}

    for category, data in keyword_categories.items():
        keywords = data['keywords']
        weight = data['weight']
        category_counts = {}

        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            count = len(re.findall(pattern, text_lower))
            if count > 0:
                weighted_count = count * weight
                category_counts[keyword] = {'count': count, 'weighted': weighted_count}
                all_keywords[keyword] = {'count': count, 'category': category, 'weighted': weighted_count}

        category_results[category] = category_counts

    return category_results, all_keywords

def calculate_keyword_score(all_keywords):
    """Calculate overall keyword strength score"""
    if not all_keywords:
        return 0

    total_weighted = sum([kw['weighted'] for kw in all_keywords.values()])
    unique_keywords = len(all_keywords)

    # Score based on weighted keyword count and diversity
    score = min((total_weighted / 50) * 70 + (unique_keywords / 30) * 30, 100)
    return round(score, 1)

def extract_experience_metrics(text):
    """Extract quantifiable achievements and metrics"""
    metrics = []

    # Patterns for percentages, numbers with metrics
    patterns = [
        r'\b(\d+)%\s*(increase|decrease|growth|reduction|improvement)',
        r'\$\s*(\d+[KMB]?)',
        r'\b(\d+)\s*(users|customers|clients|projects|teams|people|members)',
        r'\b(\d+)\+?\s*(years?|months?)\s*(of\s*)?(experience|expertise)',
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            metrics.append(match.group(0))

    return metrics[:10]  # Return top 10 metrics

def analyze_experience_level(text):
    """Determine experience level based on content"""
    text_lower = text.lower()

    # Years of experience
    years_matches = re.findall(r'(\d+)\+?\s*years?', text_lower)
    max_years = max([int(y) for y in years_matches]) if years_matches else 0

    # Job titles
    senior_titles = len(re.findall(r'\b(senior|lead|principal|director|manager|head of|vp|chief)\b', text_lower))
    entry_titles = len(re.findall(r'\b(intern|junior|associate|assistant|trainee)\b', text_lower))

    if max_years >= 7 or senior_titles >= 2:
        return "Senior (7+ years)", "expert"
    elif max_years >= 3 or (senior_titles >= 1 and entry_titles == 0):
        return "Mid-Level (3-7 years)", "intermediate"
    else:
        return "Entry-Level (0-3 years)", "beginner"

def generate_recommendations(score, sections, keyword_score, experience_level):
    """Generate personalized recommendations"""
    recommendations = []

    if score < 70:
        recommendations.append("🎯 **Priority**: Improve ATS compatibility by ensuring all text is machine-readable")

    if not sections['summary']:
        recommendations.append("📝 Add a professional summary at the top highlighting your key qualifications")

    if not sections['skills']:
        recommendations.append("💡 Create a dedicated skills section with relevant technical and soft skills")

    if keyword_score < 50:
        recommendations.append("🔑 Incorporate more industry-specific keywords and action verbs")

    if not sections['projects'] and experience_level[1] == "beginner":
        recommendations.append("🚀 Add a projects section to showcase your hands-on experience")

    if not sections['certifications']:
        recommendations.append("🏆 Consider adding relevant certifications to boost credibility")

    recommendations.append("📊 Use quantifiable metrics (%, $, numbers) to demonstrate impact")
    recommendations.append("✨ Tailor your resume for each job application using relevant keywords")

    return recommendations

# Main App Function
def main():
    # Header
    st.markdown('<p class="main-header"><span style="-webkit-text-fill-color: initial">📝</span> AI-Powered Resume Analyzer</p>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Get comprehensive insights and actionable recommendations for your resume</p>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.write("""
        This advanced tool analyzes your resume using:
        - **ATS Compatibility Scoring**
        - **Keyword Analysis by Category**
        - **Section Detection**
        - **Experience Level Assessment**
        - **Readability Metrics**
        - **Personalized Recommendations**
        """)

        st.header("📤 Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose PDF file",
            type="pdf",
            help="Upload your resume in PDF format"
        )

        st.markdown("---")
        st.markdown("💡 **Pro Tip**: Best results with text-based PDFs, not scanned images")

    # Main content
    if uploaded_file is not None:
        st.success(f"📄 Analyzing: **{uploaded_file.name}**")

        # Extract text
        with st.spinner("🔍 Extracting and analyzing content..."):
            resume_text, page_texts, page_count = extract_text_from_pdf(uploaded_file)

        # Check if extraction was successful
        if not resume_text or len(resume_text.strip()) < 50:
            st.error("❌ Failed to extract sufficient text from the PDF.")
            st.warning("""
            **Possible reasons:**
            - The PDF is image-based (scanned document) without OCR
            - The PDF is corrupted or password-protected
            - The file format is not standard PDF

            **Solutions:**
            - Try converting your resume to a text-based PDF
            - Use a PDF editor to ensure text is selectable
            - Save your resume from Word/Google Docs as PDF
            """)
            return

        # Perform all analyses
        with st.spinner("🔍 Running advanced analysis..."):
            sections = detect_sections(resume_text)
            ats_score, ats_feedback = calculate_ats_score(resume_text, sections)
            category_keywords, all_keywords = extract_advanced_keywords(resume_text)
            keyword_score = calculate_keyword_score(all_keywords)
            metrics = extract_experience_metrics(resume_text)
            experience_level, exp_code = analyze_experience_level(resume_text)
            recommendations = generate_recommendations(ats_score, sections, keyword_score, experience_level)

        st.success("✅ Analysis Complete!")

        # Overall Score Dashboard
        st.header("📊 Overall Resume Score")

        # Add container for better visibility
        with st.container():
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("ATS Score", f"{ats_score:.0f}/100",
                         delta="Good" if ats_score >= 70 else "Needs Work",
                         delta_color="normal" if ats_score >= 70 else "inverse")

            with col2:
                st.metric("Keyword Strength", f"{keyword_score:.0f}/100",
                         delta="Strong" if keyword_score >= 60 else "Weak",
                         delta_color="normal" if keyword_score >= 60 else "inverse")

            with col3:
                sections_count = sum(sections.values())
                st.metric("Sections Found", f"{sections_count}/8",
                         delta="Complete" if sections_count >= 6 else "Missing some")

            with col4:
                st.metric("Experience Level", experience_level.split('(')[0].strip())

        st.markdown("<br>", unsafe_allow_html=True)

        # Score gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = ats_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "ATS Compatibility Score", 'font': {'size': 24}},
            delta = {'reference': 70, 'increasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#ffcccb'},
                    {'range': [50, 70], 'color': '#fff4cc'},
                    {'range': [70, 100], 'color': '#ccffcc'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70}}))

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Detailed Analysis Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 ATS Analysis", "🔑 Keywords", "📑 Sections", "📈 Metrics", "💡 Recommendations"])

        with tab1:
            st.subheader("ATS Friendliness Analysis")

            # Add background for better visibility
            with st.container():
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write("**Detailed Feedback:**")
                    for item in ats_feedback:
                        st.write(item)

                with col2:
                    st.markdown('<p class="stats-header">Statistics:</p>', unsafe_allow_html=True)
                    st.metric("Text Length", f"{len(resume_text):,} chars")
                    st.metric("Page Count", str(page_count))
                    readability = textstat.flesch_reading_ease(resume_text[:1000])
                    st.metric("Readability", f"{readability:.0f}/100")

        with tab2:
            st.subheader("Keyword Analysis by Category")

            if all_keywords:
                # Category breakdown
                for category, keywords in category_keywords.items():
                    if keywords:
                        with st.expander(f"**{category}** ({len(keywords)} keywords found)"):
                            # Sort by count
                            sorted_kw = sorted(keywords.items(), key=lambda x: x[1]['count'], reverse=True)

                            # Create DataFrame
                            df = pd.DataFrame([
                                {'Keyword': k.title(), 'Count': v['count'], 'Weighted Score': f"{v['weighted']:.1f}"}
                                for k, v in sorted_kw
                            ])
                            st.dataframe(df, hide_index=True, use_container_width=True)

                # Top keywords chart
                st.subheader("Top 15 Keywords")
                sorted_all = sorted(all_keywords.items(), key=lambda x: x[1]['weighted'], reverse=True)[:15]

                keywords_df = pd.DataFrame([
                    {'Keyword': k.title(), 'Count': v['count'], 'Category': v['category']}
                    for k, v in sorted_all
                ])

                fig = px.bar(keywords_df, x='Keyword', y='Count', color='Category',
                            title="Most Frequent Keywords",
                            color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No keywords detected. Consider adding more industry-relevant terms.")

        with tab3:
            st.subheader("Resume Sections Detected")

            sections_df = pd.DataFrame([
                {'Section': k.title(), 'Present': '✅ Yes' if v else '❌ No', 'Status': 'Complete' if v else 'Missing'}
                for k, v in sections.items()
            ])

            fig = px.pie(sections_df, names='Section',
                        color='Status',
                        color_discrete_map={'Complete': '#90EE90', 'Missing': '#FFB6C6'},
                        title="Section Completeness")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(sections_df, hide_index=True, use_container_width=True)

        with tab4:
            st.subheader("Quantifiable Achievements")

            if metrics:
                st.success(f"Found {len(metrics)} quantifiable metrics in your resume!")
                st.write("**Detected Metrics:**")
                for i, metric in enumerate(metrics, 1):
                    st.write(f"{i}. {metric}")
            else:
                st.warning("⚠️ No quantifiable metrics detected!")
                st.write("Add numbers, percentages, and concrete results to make your achievements more impactful.")

            st.subheader("Experience Level Assessment")
            st.info(f"**Detected Level:** {experience_level}")

        with tab5:
            st.subheader("Personalized Recommendations")

            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")

            # Comparison chart
            st.subheader("Your Resume vs. Industry Standards")

            comparison_df = pd.DataFrame({
                'Metric': ['ATS Score', 'Keywords', 'Sections', 'Readability'],
                'Your Resume': [ats_score, keyword_score, (sum(sections.values())/8)*100,
                               min(textstat.flesch_reading_ease(resume_text[:1000]), 100)],
                'Industry Standard': [80, 70, 75, 70]
            })

            fig = go.Figure()
            fig.add_trace(go.Bar(name='Your Resume', x=comparison_df['Metric'],
                                y=comparison_df['Your Resume'], marker_color='lightblue'))
            fig.add_trace(go.Bar(name='Industry Standard', x=comparison_df['Metric'],
                                y=comparison_df['Industry Standard'], marker_color='lightcoral'))

            fig.update_layout(barmode='group', title="Performance Comparison")
            st.plotly_chart(fig, use_container_width=True)

    else:
        # Welcome screen
        st.info("👈 Upload your resume PDF using the sidebar to get started!")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🎯 ATS Scoring")
            st.write("Advanced compatibility analysis with detailed feedback")

        with col2:
            st.markdown("### 📊 Visual Analytics")
            st.write("Interactive charts and comprehensive insights")

        with col3:
            st.markdown("### 💡 Smart Tips")
            st.write("AI-powered recommendations for improvement")

if __name__ == "__main__":
    main()
