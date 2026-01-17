

"""
Advanced PDF Document Analyzer & Summarizer
A comprehensive Streamlit application for extracting, analyzing, and summarizing PDF documents.
Features: Multi-algorithm summarization, sentiment analysis, visualization, and advanced NLP metrics.
Colab-friendly version with automatic NLTK data download.
"""

import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from rake_nltk import Rake
import io
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
import json
import sys

# Configure page settings
st.set_page_config(
    page_title="Advanced PDF Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download required NLTK data with error handling
@st.cache_resource
def download_nltk_data():
    """Download all required NLTK data packages."""
    packages = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'averaged_perceptron_tagger',
        'wordnet',
        'omw-1.4'
    ]

    for package in packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except:
                pass

    # Additional check for punkt_tab specifically
    try:
        nltk.data.find('tokenizers/punkt_tab/english/')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            pass

# Initialize NLTK data
try:
    download_nltk_data()
except Exception as e:
    st.warning(f"NLTK data download issue (non-critical): {str(e)}")

# Custom CSS for professional UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
    </style>
    """, unsafe_allow_html=True)


def safe_sent_tokenize(text):
    """Safely tokenize text into sentences with fallback."""
    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        # Fallback: split by common sentence endings
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]


def safe_word_tokenize(text):
    """Safely tokenize text into words with fallback."""
    try:
        return nltk.word_tokenize(text)
    except LookupError:
        # Fallback: simple word splitting
        return re.findall(r'\b\w+\b', text.lower())


def extract_text_from_pdf(pdf_file):
    """
    Extract text content and metadata from a PDF file using PyMuPDF.

    Args:
        pdf_file: Uploaded PDF file object from Streamlit

    Returns:
        tuple: (extracted_text, metadata_dict)
    """
    try:
        pdf_bytes = pdf_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

        text = ""
        page_texts = []

        # Extract text from each page
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            page_text = page.get_text()
            text += page_text
            page_texts.append(page_text)

        # Extract metadata
        metadata = pdf_document.metadata
        metadata_dict = {
            "pages": pdf_document.page_count,
            "title": metadata.get("title", "N/A"),
            "author": metadata.get("author", "N/A"),
            "subject": metadata.get("subject", "N/A"),
            "creator": metadata.get("creator", "N/A"),
            "page_texts": page_texts
        }

        pdf_document.close()
        return text.strip(), metadata_dict

    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None, None


def clean_text(text):
    """Clean and preprocess text."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:\-\'\"]', '', text)
    return text.strip()


def generate_summary(text, num_sentences=5, algorithm="textrank"):
    """
    Generate extractive summary using various algorithms.

    Args:
        text (str): Input text to summarize
        num_sentences (int): Number of sentences in the summary
        algorithm (str): Summarization algorithm to use

    Returns:
        str: Generated summary
    """
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        stemmer = Stemmer("english")

        # Select summarizer based on algorithm
        if algorithm == "textrank":
            summarizer = TextRankSummarizer(stemmer)
        elif algorithm == "lsa":
            summarizer = LsaSummarizer(stemmer)
        elif algorithm == "luhn":
            summarizer = LuhnSummarizer(stemmer)
        else:
            summarizer = TextRankSummarizer(stemmer)

        summarizer.stop_words = get_stop_words("english")
        summary_sentences = summarizer(parser.document, num_sentences)
        summary = " ".join([str(sentence) for sentence in summary_sentences])
        return summary

    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None


def extract_keywords(text, num_keywords=10):
    """Extract top keywords using RAKE."""
    try:
        rake = Rake()
        rake.extract_keywords_from_text(text)
        keywords_with_scores = rake.get_ranked_phrases_with_scores()
        return keywords_with_scores[:num_keywords]
    except Exception as e:
        st.error(f"Error extracting keywords: {str(e)}")
        return []


def extract_tfidf_keywords(text, num_keywords=10):
    """Extract keywords using TF-IDF."""
    try:
        # Tokenize into sentences
        sentences = safe_sent_tokenize(text)

        if len(sentences) < 2:
            return []

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=num_keywords, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Get feature names and scores
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.sum(axis=0).A1

        # Create keyword-score pairs
        keywords = [(score, word) for word, score in zip(feature_names, scores)]
        keywords.sort(reverse=True)

        return keywords[:num_keywords]
    except Exception as e:
        return []


def perform_sentiment_analysis(text):
    """
    Perform sentiment analysis on the text.

    Args:
        text (str): Input text

    Returns:
        dict: Sentiment scores and classification
    """
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Classify sentiment
        if polarity > 0.1:
            sentiment = "Positive"
        elif polarity < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "sentiment": sentiment
        }
    except Exception as e:
        return None


def analyze_readability(text):
    """
    Calculate readability metrics.

    Args:
        text (str): Input text

    Returns:
        dict: Readability scores
    """
    try:
        sentences = safe_sent_tokenize(text)
        words = safe_word_tokenize(text)

        # Calculate metrics
        num_sentences = len(sentences)
        num_words = len(words)
        num_chars = sum(len(word) for word in words)

        # Avoid division by zero
        if num_sentences == 0 or num_words == 0:
            return None

        avg_words_per_sentence = num_words / num_sentences
        avg_chars_per_word = num_chars / num_words

        # Flesch Reading Ease (simplified)
        flesch_score = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_chars_per_word
        flesch_score = max(0, min(100, flesch_score))  # Clamp between 0-100

        # Interpret score
        if flesch_score >= 90:
            difficulty = "Very Easy"
        elif flesch_score >= 80:
            difficulty = "Easy"
        elif flesch_score >= 70:
            difficulty = "Fairly Easy"
        elif flesch_score >= 60:
            difficulty = "Standard"
        elif flesch_score >= 50:
            difficulty = "Fairly Difficult"
        elif flesch_score >= 30:
            difficulty = "Difficult"
        else:
            difficulty = "Very Difficult"

        return {
            "flesch_score": flesch_score,
            "difficulty": difficulty,
            "avg_words_per_sentence": avg_words_per_sentence,
            "avg_chars_per_word": avg_chars_per_word
        }
    except Exception as e:
        return None


def extract_entities(text):
    """
    Extract named entities (simplified version).

    Args:
        text (str): Input text

    Returns:
        dict: Counts of different entity types
    """
    try:
        # Simple pattern-based entity extraction
        # Numbers
        numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text)

        # Dates (basic patterns)
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', text)

        # Emails
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)

        # URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)

        # Capitalized words (potential proper nouns)
        words = text.split()
        capitalized = [word for word in words if word and word[0].isupper() and len(word) > 1]

        return {
            "numbers": len(numbers),
            "dates": len(dates),
            "emails": len(emails),
            "urls": len(urls),
            "capitalized_words": len(set(capitalized))
        }
    except Exception as e:
        return None


def calculate_word_frequency(text, top_n=20):
    """Calculate word frequency distribution."""
    try:
        # Tokenize and clean
        words = safe_word_tokenize(text)

        # Remove stopwords and punctuation
        try:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
        except:
            # Fallback stopwords list
            stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                            'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that',
                            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'])

        words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 2]

        # Count frequency
        word_freq = Counter(words)
        return word_freq.most_common(top_n)
    except Exception as e:
        return []


def generate_wordcloud(text):
    """Generate word cloud image."""
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    except Exception as e:
        return None


def calculate_text_statistics(text):
    """Calculate comprehensive text statistics."""
    try:
        sentences = safe_sent_tokenize(text)
        words = safe_word_tokenize(text)

        # Character counts
        total_chars = len(text)
        chars_no_spaces = len(text.replace(" ", ""))

        # Word statistics
        word_lengths = [len(word) for word in words if word.isalnum()]
        avg_word_length = np.mean(word_lengths) if word_lengths else 0

        # Sentence statistics
        sentence_lengths = [len(safe_word_tokenize(sent)) for sent in sentences]
        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0

        # Vocabulary richness (Type-Token Ratio)
        unique_words = set(word.lower() for word in words if word.isalnum())
        total_words = len([word for word in words if word.isalnum()])
        ttr = len(unique_words) / total_words if total_words > 0 else 0

        return {
            "total_sentences": len(sentences),
            "total_words": total_words,
            "unique_words": len(unique_words),
            "total_chars": total_chars,
            "chars_no_spaces": chars_no_spaces,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "ttr": ttr,
            "longest_sentence": max(sentence_lengths) if sentence_lengths else 0,
            "shortest_sentence": min(sentence_lengths) if sentence_lengths else 0
        }
    except Exception as e:
        return None


def create_comparison_chart(original_count, summary_count):
    """Create a comparison chart for word counts."""
    fig = go.Figure(data=[
        go.Bar(
            x=['Original Document', 'Summary'],
            y=[original_count, summary_count],
            marker_color=['#1f77b4', '#ff7f0e'],
            text=[f'{original_count:,}', f'{summary_count:,}'],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title='Document vs Summary Word Count',
        yaxis_title='Word Count',
        showlegend=False,
        height=400
    )

    return fig


def create_sentiment_gauge(polarity):
    """Create a gauge chart for sentiment."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=polarity,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sentiment Polarity"},
        delta={'reference': 0},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.1], 'color': "lightcoral"},
                {'range': [-0.1, 0.1], 'color': "lightyellow"},
                {'range': [0.1, 1], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': polarity
            }
        }
    ))

    fig.update_layout(height=300)
    return fig


def create_word_frequency_chart(word_freq):
    """Create a horizontal bar chart for word frequency."""
    if not word_freq:
        return None

    words, counts = zip(*word_freq)

    fig = go.Figure(go.Bar(
        x=counts,
        y=words,
        orientation='h',
        marker_color='#1f77b4'
    ))

    fig.update_layout(
        title='Top Words by Frequency',
        xaxis_title='Frequency',
        yaxis_title='Words',
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig


def export_analysis_report(data):
    """Export analysis data as JSON."""
    return json.dumps(data, indent=2)


def main():
    """Main application function."""

    # Header
    st.markdown('<p class="main-header"><span style="-webkit-text-fill-color: initial">📑</span> Advanced PDF Document Analyzer</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Algorithm Summarization • Sentiment Analysis • Advanced NLP Metrics • Data Visualization</p>',
                unsafe_allow_html=True)

    # Colab detection and info
    if 'COLAB_GPU' in st.session_state or 'google.colab' in sys.modules:
        st.info("🔬 Running in Google Colab environment - All dependencies configured!")

    st.markdown("---")

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration Panel")

        # File uploader
        uploaded_file = st.file_uploader(
            "📁 Upload PDF Document",
            type=["pdf"],
            help="Upload a PDF file for comprehensive analysis"
        )

        st.markdown("---")

        # Summarization settings
        st.subheader("📝 Summarization Settings")

        algorithm = st.selectbox(
            "Algorithm",
            ["textrank", "lsa", "luhn"],
            help="TextRank: Graph-based, LSA: Latent Semantic Analysis, Luhn: Frequency-based"
        )

        num_sentences = st.slider(
            "Summary Sentences",
            min_value=3,
            max_value=20,
            value=7,
            help="Number of sentences in the summary"
        )

        st.markdown("---")

        # Analysis options
        st.subheader("🔍 Analysis Options")

        show_sentiment = st.checkbox("Sentiment Analysis", value=True)
        show_readability = st.checkbox("Readability Metrics", value=True)
        show_keywords = st.checkbox("Keyword Extraction", value=True)
        show_wordcloud = st.checkbox("Word Cloud", value=True)
        show_frequency = st.checkbox("Word Frequency", value=True)

        st.markdown("---")

        # About section
        st.subheader("ℹ️ About")
        st.info(
            """
            **Advanced Features:**
            - Multi-algorithm summarization
            - Sentiment & emotion analysis
            - Readability scoring
            - TF-IDF keyword extraction
            - Entity recognition
            - Statistical analysis
            - Interactive visualizations
            - Export functionality
            """
        )

        st.markdown("---")
        st.caption(f"© 2024 | Last Updated: {datetime.now().strftime('%Y-%m-%d')}")

    # Main content area
    if uploaded_file is not None:

        # Extract text and metadata
        with st.spinner("🔄 Extracting and analyzing PDF..."):
            extracted_text, metadata = extract_text_from_pdf(uploaded_file)

            if extracted_text and metadata:
                cleaned_text = clean_text(extracted_text)

                # Check text length
                word_count = len(cleaned_text.split())

                if word_count < num_sentences * 5:
                    st.warning("⚠️ Document is too short for the selected summary length. Reduce the number of sentences.")
                    return

                # Create tabs for organized content
                tabs = st.tabs([
                    "📊 Overview",
                    "📝 Summary",
                    "🎯 Keywords",
                    "💭 Sentiment",
                    "📈 Statistics",
                    "📄 Document"
                ])

                # TAB 1: Overview
                with tabs[0]:
                    st.subheader("📋 Document Overview")

                    # Document metadata
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("📄 Pages", metadata['pages'])
                    with col2:
                        st.metric("📝 Words", f"{word_count:,}")
                    with col3:
                        st.metric("🔤 Characters", f"{len(cleaned_text):,}")
                    with col4:
                        sentences_count = len(safe_sent_tokenize(cleaned_text))
                        st.metric("📋 Sentences", sentences_count)

                    st.markdown("---")

                    # Metadata table
                    st.subheader("📚 Document Metadata")
                    metadata_df = pd.DataFrame({
                        "Property": ["Title", "Author", "Subject", "Creator", "Pages"],
                        "Value": [
                            metadata['title'],
                            metadata['author'],
                            metadata['subject'],
                            metadata['creator'],
                            metadata['pages']
                        ]
                    })
                    st.dataframe(metadata_df, use_container_width=True, hide_index=True)

                    st.markdown("---")

                    # Quick statistics
                    st.subheader("⚡ Quick Statistics")
                    stats = calculate_text_statistics(cleaned_text)

                    if stats:
                        col1, col2 = st.columns(2)

                        with col1:
                            stats_df1 = pd.DataFrame({
                                "Metric": [
                                    "Total Sentences",
                                    "Total Words",
                                    "Unique Words",
                                    "Vocabulary Richness (TTR)"
                                ],
                                "Value": [
                                    f"{stats['total_sentences']:,}",
                                    f"{stats['total_words']:,}",
                                    f"{stats['unique_words']:,}",
                                    f"{stats['ttr']:.2%}"
                                ]
                            })
                            st.dataframe(stats_df1, use_container_width=True, hide_index=True)

                        with col2:
                            stats_df2 = pd.DataFrame({
                                "Metric": [
                                    "Avg Word Length",
                                    "Avg Sentence Length",
                                    "Longest Sentence",
                                    "Shortest Sentence"
                                ],
                                "Value": [
                                    f"{stats['avg_word_length']:.1f} chars",
                                    f"{stats['avg_sentence_length']:.1f} words",
                                    f"{stats['longest_sentence']} words",
                                    f"{stats['shortest_sentence']} words"
                                ]
                            })
                            st.dataframe(stats_df2, use_container_width=True, hide_index=True)

                # TAB 2: Summary
                with tabs[1]:
                    st.subheader(f"📝 Document Summary ({algorithm.upper()} Algorithm)")

                    # Generate summary
                    with st.spinner("Generating summary..."):
                        summary = generate_summary(cleaned_text, num_sentences, algorithm)

                    if summary:
                        st.success("✅ Summary generated successfully!")

                        # Display summary in a nice box
                        st.markdown("### Summary Text")
                        st.info(summary)

                        # Summary metrics
                        st.markdown("---")
                        st.subheader("📊 Summary Metrics")

                        summary_word_count = len(summary.split())
                        compression_ratio = (summary_word_count / word_count) * 100

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Summary Words", f"{summary_word_count:,}")
                        with col2:
                            st.metric("Compression Ratio", f"{compression_ratio:.1f}%")
                        with col3:
                            time_saved = ((word_count - summary_word_count) / 200)  # 200 WPM
                            st.metric("Time Saved", f"{time_saved:.1f} min")

                        # Comparison chart
                        st.plotly_chart(
                            create_comparison_chart(word_count, summary_word_count),
                            use_container_width=True
                        )

                        # Download options
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="📥 Download Summary (TXT)",
                                data=summary,
                                file_name=f"summary_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                        with col2:
                            # Create detailed report
                            report = f"""DOCUMENT SUMMARY REPORT
                            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                            Algorithm: {algorithm.upper()}
                            Original Words: {word_count:,}
                            Summary Words: {summary_word_count:,}
                            Compression: {compression_ratio:.1f}%

                            SUMMARY:
                            {summary}
                            """
                            st.download_button(
                                label="📥 Download Report (TXT)",
                                data=report,
                                file_name=f"report_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )

                # TAB 3: Keywords
                with tabs[2]:
                    if show_keywords:
                        st.subheader("🎯 Keyword Analysis")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("### 🔑 RAKE Keywords")
                            rake_keywords = extract_keywords(cleaned_text, num_keywords=10)

                            if rake_keywords:
                                rake_df = pd.DataFrame({
                                    "Rank": range(1, len(rake_keywords) + 1),
                                    "Keyword": [kw[1] for kw in rake_keywords],
                                    "Score": [f"{kw[0]:.2f}" for kw in rake_keywords]
                                })
                                st.dataframe(rake_df, use_container_width=True, hide_index=True)

                        with col2:
                            st.markdown("### 📊 TF-IDF Keywords")
                            tfidf_keywords = extract_tfidf_keywords(cleaned_text, num_keywords=10)

                            if tfidf_keywords:
                                tfidf_df = pd.DataFrame({
                                    "Rank": range(1, len(tfidf_keywords) + 1),
                                    "Keyword": [kw[1] for kw in tfidf_keywords],
                                    "TF-IDF Score": [f"{kw[0]:.4f}" for kw in tfidf_keywords]
                                })
                                st.dataframe(tfidf_df, use_container_width=True, hide_index=True)

                        # Word cloud
                        if show_wordcloud:
                            st.markdown("---")
                            st.markdown("### ☁️ Word Cloud Visualization")

                            wordcloud_fig = generate_wordcloud(cleaned_text)
                            if wordcloud_fig:
                                st.pyplot(wordcloud_fig)
                    else:
                        st.info("Enable 'Keyword Extraction' in the sidebar to view this analysis.")

                # TAB 4: Sentiment
                with tabs[3]:
                    if show_sentiment:
                        st.subheader("💭 Sentiment Analysis")

                        sentiment_data = perform_sentiment_analysis(cleaned_text)

                        if sentiment_data:
                            col1, col2 = st.columns([1, 1])

                            with col1:
                                st.markdown("### 📊 Sentiment Metrics")

                                # Display sentiment badge
                                sentiment = sentiment_data['sentiment']
                                if sentiment == "Positive":
                                    st.success(f"**Overall Sentiment:** {sentiment} 😊")
                                elif sentiment == "Negative":
                                    st.error(f"**Overall Sentiment:** {sentiment} 😞")
                                else:
                                    st.info(f"**Overall Sentiment:** {sentiment} 😐")

                                # Metrics
                                sentiment_df = pd.DataFrame({
                                    "Metric": ["Polarity", "Subjectivity", "Classification"],
                                    "Value": [
                                        f"{sentiment_data['polarity']:.3f}",
                                        f"{sentiment_data['subjectivity']:.3f}",
                                        sentiment_data['sentiment']
                                    ],
                                    "Description": [
                                        "Range: -1 (negative) to +1 (positive)",
                                        "Range: 0 (objective) to 1 (subjective)",
                                        "Overall sentiment category"
                                    ]
                                })
                                st.dataframe(sentiment_df, use_container_width=True, hide_index=True)

                            with col2:
                                st.markdown("### 🎚️ Sentiment Gauge")
                                sentiment_gauge = create_sentiment_gauge(sentiment_data['polarity'])
                                st.plotly_chart(sentiment_gauge, use_container_width=True)

                            # Interpretation
                            st.markdown("---")
                            st.markdown("### 📖 Interpretation")

                            interpretation = f"""
                            **Polarity Analysis:** The document has a polarity score of {sentiment_data['polarity']:.3f},
                            indicating a {sentiment.lower()} tone overall. Scores closer to +1 indicate positive sentiment,
                            while scores closer to -1 indicate negative sentiment.

                            **Subjectivity Analysis:** With a subjectivity score of {sentiment_data['subjectivity']:.3f},
                            the document is {'mostly subjective (opinion-based)' if sentiment_data['subjectivity'] > 0.5 else 'mostly objective (fact-based)'}.
                            Scores closer to 1 indicate more personal opinions, while scores closer to 0 indicate more factual content.
                            """
                            st.markdown(interpretation)
                    else:
                        st.info("Enable 'Sentiment Analysis' in the sidebar to view this analysis.")

                # TAB 5: Statistics
                with tabs[4]:
                    st.subheader("📈 Advanced Statistics")

                    # Readability analysis
                    if show_readability:
                        st.markdown("### 📚 Readability Analysis")

                        readability = analyze_readability(cleaned_text)

                        if readability:
                            col1, col2 = st.columns(2)

                            with col1:
                                # Flesch Reading Ease score with color coding
                                flesch_score = readability['flesch_score']
                                if flesch_score >= 60:
                                    st.success(f"**Flesch Reading Ease:** {flesch_score:.1f}")
                                elif flesch_score >= 30:
                                    st.warning(f"**Flesch Reading Ease:** {flesch_score:.1f}")
                                else:
                                    st.error(f"**Flesch Reading Ease:** {flesch_score:.1f}")

                                st.info(f"**Difficulty Level:** {readability['difficulty']}")

                                # Create a progress bar for readability
                                st.progress(flesch_score / 100)

                            with col2:
                                readability_df = pd.DataFrame({
                                    "Metric": [
                                        "Avg Words per Sentence",
                                        "Avg Characters per Word",
                                        "Flesch Score",
                                        "Difficulty Level"
                                    ],
                                    "Value": [
                                        f"{readability['avg_words_per_sentence']:.1f}",
                                        f"{readability['avg_chars_per_word']:.1f}",
                                        f"{readability['flesch_score']:.1f}",
                                        readability['difficulty']
                                    ]
                                })
                                st.dataframe(readability_df, use_container_width=True, hide_index=True)

                            # Readability interpretation
                            st.markdown("---")
                            st.markdown("**📖 Readability Guide:**")
                            st.markdown("""
                            - **90-100:** Very Easy (5th grade)
                            - **80-89:** Easy (6th grade)
                            - **70-79:** Fairly Easy (7th grade)
                            - **60-69:** Standard (8th-9th grade)
                            - **50-59:** Fairly Difficult (10th-12th grade)
                            - **30-49:** Difficult (College level)
                            - **0-29:** Very Difficult (College graduate)
                            """)

                    st.markdown("---")

                    # Word frequency analysis
                    if show_frequency:
                        st.markdown("### 📊 Word Frequency Distribution")

                        word_freq = calculate_word_frequency(cleaned_text, top_n=20)

                        if word_freq:
                            freq_chart = create_word_frequency_chart(word_freq)
                            if freq_chart:
                                st.plotly_chart(freq_chart, use_container_width=True)

                            # Display as table as well
                            with st.expander("📋 View Frequency Table"):
                                freq_df = pd.DataFrame(word_freq, columns=["Word", "Frequency"])
                                freq_df.index = range(1, len(freq_df) + 1)
                                st.dataframe(freq_df, use_container_width=True)

                    st.markdown("---")

                    # Entity analysis
                    st.markdown("### 🏷️ Entity Detection")

                    entities = extract_entities(cleaned_text)

                    if entities:
                        entity_col1, entity_col2, entity_col3 = st.columns(3)

                        with entity_col1:
                            st.metric("🔢 Numbers Found", entities['numbers'])
                            st.metric("📅 Dates Found", entities['dates'])

                        with entity_col2:
                            st.metric("📧 Emails Found", entities['emails'])
                            st.metric("🔗 URLs Found", entities['urls'])

                        with entity_col3:
                            st.metric("🏢 Proper Nouns", entities['capitalized_words'])

                    st.markdown("---")

                    # Detailed statistics table
                    st.markdown("### 📋 Comprehensive Statistics")

                    stats = calculate_text_statistics(cleaned_text)

                    if stats:
                        detailed_stats_df = pd.DataFrame({
                            "Category": [
                                "Document Length",
                                "Document Length",
                                "Document Length",
                                "Document Length",
                                "Vocabulary",
                                "Vocabulary",
                                "Vocabulary",
                                "Sentence Analysis",
                                "Sentence Analysis",
                                "Sentence Analysis"
                            ],
                            "Metric": [
                                "Total Characters",
                                "Characters (no spaces)",
                                "Total Words",
                                "Total Sentences",
                                "Unique Words",
                                "Vocabulary Richness (TTR)",
                                "Average Word Length",
                                "Average Sentence Length",
                                "Longest Sentence",
                                "Shortest Sentence"
                            ],
                            "Value": [
                                f"{stats['total_chars']:,}",
                                f"{stats['chars_no_spaces']:,}",
                                f"{stats['total_words']:,}",
                                f"{stats['total_sentences']:,}",
                                f"{stats['unique_words']:,}",
                                f"{stats['ttr']:.2%}",
                                f"{stats['avg_word_length']:.1f} characters",
                                f"{stats['avg_sentence_length']:.1f} words",
                                f"{stats['longest_sentence']} words",
                                f"{stats['shortest_sentence']} words"
                            ]
                        })
                        st.dataframe(detailed_stats_df, use_container_width=True, hide_index=True)

                # TAB 6: Document
                with tabs[5]:
                    st.subheader("📄 Document Content")

                    # Page-by-page analysis
                    if metadata['pages'] > 1:
                        st.markdown("### 📑 Page Analysis")

                        page_data = []
                        for i, page_text in enumerate(metadata['page_texts'], 1):
                            page_words = len(page_text.split())
                            page_chars = len(page_text)
                            page_data.append({
                                "Page": i,
                                "Words": page_words,
                                "Characters": page_chars
                            })

                        page_df = pd.DataFrame(page_data)

                        # Create page statistics chart
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=page_df['Page'],
                            y=page_df['Words'],
                            name='Words per Page',
                            marker_color='#1f77b4'
                        ))

                        fig.update_layout(
                            title='Words Distribution Across Pages',
                            xaxis_title='Page Number',
                            yaxis_title='Word Count',
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Page statistics table
                        with st.expander("📊 View Page Statistics Table"):
                            st.dataframe(page_df, use_container_width=True, hide_index=True)

                    st.markdown("---")

                    # Display full text
                    st.markdown("### 📖 Full Document Text")

                    # Text display options
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        display_option = st.radio(
                            "Display Mode:",
                            ["Formatted", "Raw"],
                            horizontal=True
                        )

                    with col2:
                        show_lines = st.checkbox("Show Line Numbers", value=False)

                    # Display text based on options
                    if display_option == "Formatted":
                        if show_lines:
                            lines = cleaned_text.split('\n')
                            numbered_text = '\n'.join([f"{i+1}: {line}" for i, line in enumerate(lines)])
                            st.text_area("Document Content", numbered_text, height=500)
                        else:
                            st.text_area("Document Content", cleaned_text, height=500)
                    else:
                        st.text_area("Document Content (Raw)", extracted_text, height=500)

                    # Download original text
                    st.download_button(
                        label="📥 Download Full Text",
                        data=cleaned_text,
                        file_name=f"extracted_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )

                # Export functionality (bottom of page)
                st.markdown("---")
                st.subheader("💾 Export Analysis")

                col1, col2, col3 = st.columns(3)

                with col1:
                    # Prepare comprehensive report
                    if st.button("📊 Generate Full Report", use_container_width=True):
                        with st.spinner("Generating comprehensive report..."):
                            report_data = {
                                "metadata": {
                                    "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    "filename": uploaded_file.name,
                                    "pages": metadata['pages']
                                },
                                "statistics": stats,
                                "sentiment": sentiment_data if show_sentiment else None,
                                "readability": readability if show_readability else None,
                                "keywords_rake": [(kw[1], kw[0]) for kw in rake_keywords] if show_keywords else None,
                                "keywords_tfidf": [(kw[1], kw[0]) for kw in tfidf_keywords] if show_keywords else None,
                                "entities": entities,
                                "summary": {
                                    "algorithm": algorithm,
                                    "text": summary,
                                    "compression_ratio": compression_ratio
                                }
                            }

                            json_report = export_analysis_report(report_data)

                            st.download_button(
                                label="📥 Download JSON Report",
                                data=json_report,
                                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True
                            )

                with col2:
                    # Export statistics as CSV
                    if st.button("📈 Export Statistics CSV", use_container_width=True):
                        # Combine all statistics into one DataFrame
                        export_df = pd.DataFrame({
                            "Metric": [
                                "Total Pages",
                                "Total Words",
                                "Total Characters",
                                "Unique Words",
                                "Vocabulary Richness",
                                "Avg Word Length",
                                "Avg Sentence Length",
                                "Flesch Score",
                                "Sentiment Polarity",
                                "Sentiment Subjectivity",
                                "Summary Compression Ratio"
                            ],
                            "Value": [
                                metadata['pages'],
                                stats['total_words'] if stats else 0,
                                stats['total_chars'] if stats else 0,
                                stats['unique_words'] if stats else 0,
                                f"{stats['ttr']:.2%}" if stats else "N/A",
                                f"{stats['avg_word_length']:.1f}" if stats else "N/A",
                                f"{stats['avg_sentence_length']:.1f}" if stats else "N/A",
                                f"{readability['flesch_score']:.1f}" if readability else "N/A",
                                f"{sentiment_data['polarity']:.3f}" if sentiment_data else "N/A",
                                f"{sentiment_data['subjectivity']:.3f}" if sentiment_data else "N/A",
                                f"{compression_ratio:.1f}%" if summary else "N/A"
                            ]
                        })

                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                with col3:
                    # Export keywords
                    if st.button("🔑 Export Keywords", use_container_width=True):
                        if show_keywords and rake_keywords:
                            keywords_export_df = pd.DataFrame({
                                "Rank": range(1, len(rake_keywords) + 1),
                                "Keyword": [kw[1] for kw in rake_keywords],
                                "RAKE Score": [kw[0] for kw in rake_keywords]
                            })

                            csv = keywords_export_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Keywords CSV",
                                data=csv,
                                file_name=f"keywords_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        else:
                            st.warning("Enable keyword extraction first!")

            else:
                st.error("❌ Failed to extract text from PDF. Please ensure the file is valid and not encrypted.")

    else:
        # Welcome screen
        st.info("👈 **Get Started:** Upload a PDF document using the sidebar to begin comprehensive analysis.")

        st.markdown("---")

        # Feature showcase
        st.markdown("## 🚀 Key Features")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("### 📝 Summarization")
            st.markdown("""
            - TextRank Algorithm
            - LSA (Latent Semantic Analysis)
            - Luhn Algorithm
            - Customizable length
            - Compression metrics
            """)

        with col2:
            st.markdown("### 🎯 Keywords")
            st.markdown("""
            - RAKE extraction
            - TF-IDF analysis
            - Word cloud visualization
            - Frequency analysis
            - Entity recognition
            """)

        with col3:
            st.markdown("### 💭 Sentiment")
            st.markdown("""
            - Polarity analysis
            - Subjectivity scoring
            - Visual gauge charts
            - Detailed interpretation
            - Emotion detection
            """)

        with col4:
            st.markdown("### 📊 Statistics")
            st.markdown("""
            - Readability metrics
            - Vocabulary analysis
            - Text statistics
            - Page-by-page breakdown
            - Export functionality
            """)

        st.markdown("---")

        # Sample use cases
        st.markdown("## 💼 Use Cases")

        use_cases = {
            "Research Papers": "Quickly understand academic papers and extract key findings",
            "Legal Documents": "Analyze contracts and legal texts for readability and key terms",
            "Business Reports": "Summarize lengthy reports and identify important metrics",
            "Technical Documentation": "Extract key information from technical manuals",
            "News Articles": "Get quick summaries and sentiment analysis of news content",
            "Literature": "Analyze writing style, vocabulary, and themes"
        }

        for use_case, description in use_cases.items():
            st.markdown(f"**{use_case}:** {description}")

        st.markdown("---")

        # Technical details
        with st.expander("🔧 Technical Details"):
            st.markdown("""
            ### Algorithms & Libraries Used:

            **Text Extraction:**
            - PyMuPDF (fitz) for robust PDF parsing

            **Summarization:**
            - TextRank: Graph-based ranking algorithm
            - LSA: Singular Value Decomposition for semantic analysis
            - Luhn: Statistical word frequency method

            **NLP Analysis:**
            - RAKE (Rapid Automatic Keyword Extraction)
            - TF-IDF (Term Frequency-Inverse Document Frequency)
            - TextBlob for sentiment analysis
            - NLTK for tokenization and text processing

            **Visualization:**
            - Plotly for interactive charts
            - Matplotlib for word clouds
            - Streamlit for UI components

            **Metrics:**
            - Flesch Reading Ease Score
            - Type-Token Ratio (TTR)
            - Sentiment polarity & subjectivity
            - Compression ratios
            """)

        st.markdown("---")

        # Colab setup instructions
        with st.expander("🔬 Google Colab Setup Instructions"):
            st.markdown("""
            ### Running in Google Colab:

            1. **Install Streamlit and dependencies:**
            ```python
            !pip install streamlit pandas numpy PyMuPDF sumy rake-nltk nltk textblob scikit-learn plotly wordcloud matplotlib -q
            ```

            2. **Download NLTK data:**
            ```python
            import nltk
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('stopwords')
            nltk.download('averaged_perceptron_tagger')
            ```

            3. **Install pyngrok for tunneling:**
            ```python
            !pip install pyngrok -q
            ```

            4. **Save the app.py file and run:**
            ```python
            from pyngrok import ngrok
            import subprocess

            # Start ngrok tunnel
            public_url = ngrok.connect(8501)
            print(f'Public URL: {public_url}')

            # Run Streamlit
            !streamlit run app.py --server.port 8501
            ```

            The application will be accessible via the public URL provided by ngrok!
            """)

        st.markdown("---")

        # Footer
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p><strong>Advanced PDF Document Analyzer</strong></p>
            <p>Built with Streamlit • Powered by Advanced NLP Algorithms</p>
            <p>Perfect for Data Analysts, Researchers, and Content Professionals</p>
            <p>✅ Colab-Friendly • ✅ Production-Ready • ✅ Fully Featured</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
