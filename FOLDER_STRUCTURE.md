# 📁 Folder Structure

\`\`\`
polymath-collection/
│
├── main_app.py                      # Portfolio landing page with metrics table
│
├── pages/                           # Streamlit multi-page apps folder
│   ├── 1_Data_Analyst.py        # Tools overview page
│   ├── 2_resume_analyzer.py        # Resume Analyzer (from app.py)
│   ├── 3_universal_analytics.py    # Universal Analytics (from app2.py)
│   ├── 4_csv_analyst.py            # CSV Q&A Analyst (from app3.py)
│   └── 5_pdf_analyzer.py           # PDF Analyzer (from app4.py)
│
├── .streamlit/                      # Streamlit configuration
│   └── config.toml                 # Theme and server settings
│
├── requirements.txt                 # Python dependencies
├── nltk.txt                        # NLTK data packages
├── packages.txt                    # System dependencies
│
├── README.md                       # Portfolio documentation
├── DEPLOYMENT_GUIDE.md             # Deployment instructions
├── .gitignore                      # Git ignore rules
│
└── (Optional - DO NOT COMMIT)
    ├── test_files/                 # Local test data
    └── .env                        # Local environment variables
\`\`\`

## File Purposes

### Core Application Files

**main_app.py**
- Portfolio landing page
- "The Polymath Collection" branding
- Professional metrics table
- Navigation to tools overview

**pages/1_Data_Analyst.py**
- Overview of 4 analysis tools
- Project cards with descriptions
- Navigation to individual tools

**pages/2-5: Individual Tools**
- Complete, standalone applications
- Each has own functionality
- Consistent dark theme styling

### Configuration Files

**.streamlit/config.toml**
- Dark theme colors (#8b7fd4 purple)
- Upload size limits (200MB)
- Server settings

**requirements.txt**
- All Python package dependencies
- Specific versions for stability

**nltk.txt**
- NLTK data packages to download
- Required for NLP functionality

**packages.txt**
- System-level dependencies
- Currently: libffi-dev

### Documentation Files

**README.md**
- Portfolio overview
- Installation instructions
- Technology stack
- Project structure

**DEPLOYMENT_GUIDE.md**
- Step-by-step deployment
- Troubleshooting tips
- Update procedures

**.gitignore**
- Excludes temporary files
- Protects sensitive data
- Keeps repo clean

## File Size Considerations

**Keep Individual Files Under:**
- Python files: < 1MB each
- Total repo: < 100MB recommended
- Upload size: < 200MB (Streamlit limit)

**Large Files Strategy:**
- Don't commit sample PDFs/CSVs
- Use .gitignore for test data
- Generate sample data in code instead
