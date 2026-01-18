# 🚀 The Polymath Collection

<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://the-polymath-collection.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/Hat-GitBot/The-Polymath-Collection)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

**A comprehensive suite of professional data analysis and AI-powered tools**

[Live Demo](https://the-polymath-collection.streamlit.app/) • [GitHub Repository](https://github.com/Hat-GitBot/The-Polymath-Collection) • [Report Bug](https://github.com/Hat-GitBot/The-Polymath-Collection/issues) 

</div>

## 🎯 Overview

**The Polymath Collection** is an all-in-one data science platform featuring 4 powerful, production-ready tools for data analysis, document processing, and professional insights. Built with Streamlit and powered by advanced machine learning and NLP algorithms, this application provides enterprise-grade analytics accessible through an intuitive web interface.

### Why The Polymath Collection?

- ✅ **No Installation Required** - Access all tools via web browser
- ✅ **Professional Grade** - Enterprise-level algorithms and analysis
- ✅ **User Friendly** - Intuitive interfaces with clear instructions
- ✅ **Comprehensive** - Multiple specialized tools in one platform
- ✅ **Export Ready** - Download reports, visualizations, and insights

---

## ✨ Features

### 🎨 Modern Dark Theme UI
- Sleek, professional dark interface
- Smooth animations and transitions
- Responsive design for all devices
- Intuitive navigation system

### 📊 Advanced Analytics
- Statistical analysis and hypothesis testing
- Machine learning-powered insights
- Natural language processing
- Interactive data visualizations

### 📄 Multiple Export Formats
- Word documents (.docx)
- CSV exports
- Interactive charts (Plotly)
- Downloadable reports

### 🔒 Privacy & Security
- All processing happens in-session
- No data storage on servers
- Secure file handling
- Session-based analysis

---

## 🌐 Live Application

### 🚀 **[Launch The Polymath Collection](https://the-polymath-collection.streamlit.app/)**

Access the live application instantly - no installation required!

---

## 🛠 Tools Included

### 1️⃣ **📋 Advanced CSV Data Analyst**
*Ask questions in natural language and get instant insights*

**Features:**
- 💬 Natural language Q&A interface
- 📊 Statistical analysis (mean, median, correlation, etc.)
- 🔍 Advanced filtering and grouping
- 📈 Interactive Plotly visualizations
- 📁 Multi-file support
- 🕒 Query history tracking

**Use Cases:**
- Business intelligence reporting
- Sales data analysis
- Customer behavior insights
- Performance metrics tracking

**Example Queries:**
```
"What is the average revenue by region?"
"Show me the top 10 customers by sales"
"How many orders were placed last month?"
"What's the correlation between price and quantity?"
```

---

### 2️⃣ **📝 AI Resume Analyzer**
*Professional ATS scoring and career insights*

**Features:**
- 🎯 ATS (Applicant Tracking System) compatibility scoring
- 🔑 Keyword analysis by category (Technical, Soft Skills, Business, etc.)
- 📊 Sentiment and readability metrics
- 📈 Experience level assessment
- 💡 Personalized recommendations
- 📉 Industry benchmark comparisons

**Analysis Categories:**
- Technical Skills Detection
- Action Verbs Usage
- Soft Skills Identification
- Business Keywords
- Certifications & Qualifications

**Metrics Provided:**
- ATS Score (0-100)
- Keyword Strength Score
- Section Completeness
- Readability Level (Flesch Score)
- Quantifiable Achievements Count

---

### 3️⃣ **📊 Universal Data Analytics**
*Works with ANY CSV structure - zero configuration*

**Features:**
- 🔄 Automatic column type detection
- 🎨 Smart visualization generation
- 📋 Comprehensive statistical summaries
- 📄 Professional Word report export
- 🔬 Correlation analysis
- 📊 Distribution analysis
- 🎯 Outlier detection

**What Makes It Universal:**
- No schema definition needed
- Handles any number of columns
- Automatic numerical/categorical classification
- Intelligent feature engineering
- Adaptive visualization selection

**Analysis Components:**
- Dataset overview dashboard
- Numerical distributions
- Categorical breakdowns
- Correlation heatmaps
- Box plots for outlier detection
- Scatter plots for relationships
- Automated insights and recommendations

---

### 4️⃣ **📑 PDF Document Analyzer**
*Multi-algorithm summarization and comprehensive NLP*

**Features:**
- 📝 Multiple summarization algorithms:
  - **TextRank**: Graph-based ranking
  - **LSA**: Latent Semantic Analysis
  - **Luhn**: Frequency-based method
- 💭 Sentiment analysis (polarity & subjectivity)
- 🔑 Keyword extraction (RAKE & TF-IDF)
- 📊 Readability metrics (Flesch Reading Ease)
- 📈 Text statistics and analytics
- ☁️ Word cloud generation
- 📄 Entity recognition

**Advanced Capabilities:**
- Page-by-page analysis
- Vocabulary richness measurement
- Document metadata extraction
- Named entity detection
- Compression ratio calculation
- Export comprehensive JSON reports

---

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/Hat-GitBot/The-Polymath-Collection.git
cd The-Polymath-Collection
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data** (for PDF Analyzer)
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

5. **Run the application**
```bash
streamlit run Portfolio_Dashboard.py
```

6. **Open in browser**
```
Navigate to: http://localhost:8501
```

---

## 📖 Usage

### Quick Start Guide

1. **Launch the Application**
   - Visit [https://the-polymath-collection.streamlit.app/](https://the-polymath-collection.streamlit.app/)
   - Or run locally: `streamlit run main_app.py`

2. **Choose Your Tool**
   - Navigate using the sidebar
   - Select the tool that fits your needs

3. **Upload Your Data**
   - CSV files for data analysis tools
   - PDF files for resume/document analysis

4. **Get Insights**
   - Interactive analysis results
   - Professional visualizations
   - Downloadable reports

### Example Workflows

#### CSV Data Analysis
```
1. Upload CSV file(s)
2. Ask questions: "What is the average sales by month?"
3. View statistical summaries
4. Generate custom visualizations
5. Export results
```

#### Resume Analysis
```
1. Upload PDF resume
2. View ATS compatibility score
3. Review keyword analysis
4. Check readability metrics
5. Read personalized recommendations
```

#### Universal Analytics
```
1. Upload any CSV file
2. Automatic analysis starts
3. Review comprehensive visualizations
4. Download Word report with all insights
```

#### PDF Document Analysis
```
1. Upload PDF document
2. Select summarization algorithm
3. View generated summary
4. Explore keyword extraction
5. Analyze sentiment and readability
```

---

## 🔧 Technology Stack

### Core Framework
- **Streamlit** - Web application framework
- **Python 3.8+** - Programming language

### Data Processing
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **SciPy** - Scientific computing

### Visualization
- **Plotly** - Interactive charts and graphs
- **Matplotlib** - Static visualizations
- **Seaborn** - Statistical data visualization

### Natural Language Processing
- **NLTK** - Natural language toolkit
- **TextBlob** - Sentiment analysis
- **Textstat** - Readability metrics
- **RAKE-NLTK** - Keyword extraction
- **Sumy** - Text summarization (TextRank, LSA, Luhn)
- **WordCloud** - Word cloud generation

### Document Processing
- **PyMuPDF (fitz)** - PDF text extraction
- **python-docx** - Word document generation
- **openpyxl** - Excel file handling

### Machine Learning
- **scikit-learn** - ML algorithms and tools
- **TF-IDF Vectorizer** - Text feature extraction

---

## 📁 Project Structure

```
The-Polymath-Collection/
│
├── Portfolio_Dashboard.py                        # Main landing page
│
├── pages/                               # Multi-page app directory
│   ├── 1_📋_CSV_Analyzer.py            # CSV Q&A tool
│   ├── 2_📝_Resume_Analyzer.py         # Resume analysis tool
│   ├── 3_📊_Universal_Analytics.py     # Universal CSV analytics
│   └── 4_📑_PDF_Analyzer.py            # PDF document analyzer
│
├── requirements.txt                     # Python dependencies
├── README.md                           # Project documentation
├── LICENSE                             # MIT License
│
└── .streamlit/                         # Streamlit configuration
    └── config.toml                     # App configuration
```

---

## 🤝 Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

### How to Contribute

1. **Fork the Project**
2. **Create your Feature Branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your Changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the Branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comments for complex logic
- Update documentation for new features
- Test thoroughly before submitting PR
- Keep commits atomic and well-described

---

## 📧 Contact

**Project Maintainer:** Hat-GitBot

- **GitHub:** [@Hat-GitBot](https://github.com/Hat-GitBot)
- **Project Link:** [https://github.com/Hat-GitBot/The-Polymath-Collection](https://github.com/Hat-GitBot/The-Polymath-Collection)
- **Live Application:** [https://the-polymath-collection.streamlit.app/](https://the-polymath-collection.streamlit.app/)

---

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) - For the amazing framework
- [Plotly](https://plotly.com/) - For interactive visualizations
- [NLTK](https://www.nltk.org/) - For NLP capabilities
- [PyMuPDF](https://pymupdf.readthedocs.io/) - For PDF processing
- All open-source contributors whose libraries made this possible

---

<div align="center">

**Made with ❤️ by Hat-GitBot**

[⬆ Back to Top](#-the-polymath-collection)

</div>
