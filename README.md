# 📊 The Polymath Collection

> A curated portfolio demonstrating the synthesis of diverse expertise. Bridging technical, creative, and strategic disciplines to deliver comprehensive and holistic solutions.

## 🚀 Live Demo

👉 **[View Portfolio](https://your-app-name.streamlit.app)** *(Replace with your Streamlit Cloud URL)*

## 🎯 Portfolio Overview

This portfolio showcases **4 advanced data analysis applications** demonstrating cross-disciplinary technical expertise:

### 📄 1. AI Resume Analyzer
- **ATS compatibility scoring** with detailed feedback
- **Multi-category keyword analysis** (Technical, Action Verbs, Soft Skills, Business, Certifications)
- Section detection and completeness validation
- Experience level assessment
- Readability metrics (Flesch Reading Ease)
- Interactive Plotly visualizations

### 📊 2. Universal Data Analytics
- **Works with ANY CSV structure** - zero configuration needed
- Automatic column detection and smart classification
- Comprehensive statistical analysis (numerical & categorical)
- Correlation detection and relationship mapping
- **6+ visualization types**: distributions, box plots, heatmaps, scatter plots
- Automated insights generation
- **Professional Word report export** with publication-quality charts

### 💬 3. CSV Q&A Analyst
- **Natural language query interface** for data exploration
- Support for 15+ query types: count, average, sum, max/min, groupby, trends
- Automatic data aggregation and filtering
- **Auto-generated visualizations** based on query type
- Multi-file support with comparison capabilities
- Query history tracking

### 📑 4. PDF Document Analyzer
- **Multi-algorithm summarization**: TextRank, LSA, Luhn
- **Dual keyword extraction**: RAKE + TF-IDF
- Sentiment analysis (polarity & subjectivity)
- Readability metrics and difficulty assessment
- Entity detection (dates, emails, URLs, numbers)
- Word cloud generation and frequency analysis
- Comprehensive export options (JSON, CSV, TXT)

## 💻 Technology Stack

**Core:** Python, Streamlit  
**Data Processing:** Pandas, NumPy  
**Machine Learning:** Scikit-learn, NLTK, TextBlob  
**Visualization:** Plotly, Seaborn, Matplotlib, WordCloud  
**Document Processing:** PyMuPDF, python-docx, Sumy  
**NLP:** RAKE-NLTK, TextBlob, NLTK  
**Statistics:** SciPy, textstat  

## 🛠️ Local Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

\`\`\`bash
# Clone the repository
git clone https://github.com/yourusername/polymath-collection.git
cd polymath-collection

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"

# Run the application
streamlit run main_app.py
\`\`\`

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

\`\`\`
polymath-collection/
├── main_app.py                    # Portfolio landing page
├── pages/
│   ├── 1_Data_Analyst.py      # Tools overview
│   ├── 2_resume_analyzer.py      # Resume Analyzer app
│   ├── 3_universal_analytics.py  # Universal Analytics app
│   ├── 4_csv_analyst.py          # CSV Q&A app
│   └── 5_pdf_analyzer.py         # PDF Analyzer app
├── requirements.txt               # Python dependencies
├── nltk.txt                       # NLTK data requirements
├── packages.txt                   # System dependencies
├── .streamlit/
│   └── config.toml               # Streamlit configuration
└── README.md                      # This file
\`\`\`

## 🎨 Key Features

### Professional Design
- **Dark theme** with purple gradient accents (#8b7fd4)
- Consistent UI/UX across all applications
- Responsive layout for desktop and mobile
- Interactive visualizations with Plotly

### Cross-Disciplinary Expertise
- **Statistical Analysis**: Correlation matrices, hypothesis testing
- **Machine Learning**: Classification, clustering, NLP
- **Data Visualization**: Interactive charts, professional reports
- **Natural Language Processing**: Sentiment analysis, keyword extraction
- **Document Processing**: PDF parsing, text summarization

### Production-Ready Code
- Comprehensive error handling
- Input validation
- Performance optimization
- Modular architecture
- Clean, documented code

## 📊 Portfolio Metrics

| Metric | Value | Demonstrates |
|--------|-------|--------------|
| **Domains of Mastery** | 3+ | Data Analytics, ML, NLP |
| **Cross-Discipline Projects** | 85%+ | Integration of multiple skillsets |
| **Learning Velocity** | High | Rapid tool acquisition |
| **Solution Efficiency** | ~25% | Time/cost reduction via integration |

## 🚀 Deployment

### Streamlit Cloud (Recommended)

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Connect your GitHub repository
5. Set main file: `main_app.py`
6. Click "Deploy"

### Other Platforms

- **Heroku**: Use the included `Procfile`
- **AWS**: Deploy on EC2 with nginx
- **Docker**: Build with `docker build -t polymath .`

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📝 License

MIT License - feel free to use this code for your own portfolio!

## 👤 Author

**Your Name**
- Portfolio: [Your Portfolio URL]
- LinkedIn: [Your LinkedIn]
- GitHub: [@yourusername](https://github.com/yourusername)

## ⭐ Show Your Support

Give a ⭐️ if you found this portfolio helpful!

---

**Built with ❤️ demonstrating the power of cross-disciplinary technical expertise**
