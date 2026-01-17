
import streamlit as st
import pandas as pd
import re
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Advanced CSV Data Analyst", layout="wide", page_icon="📊")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header"><span style="-webkit-text-fill-color: initial">📋</span> Advanced CSV Data Analyst</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Analysis mode
    analysis_mode = st.selectbox(
        "Analysis Mode",
        ["Basic Q&A", "Statistical Analysis", "Visual Analytics", "Data Comparison"]
    )

    st.markdown("---")
    st.header("📖 Guide")
    st.markdown("""
    **Query Examples:**
    - Count: *"How many rows?"*
    - Statistics: *"Average age"*
    - Filtering: *"Users in France"*
    - Grouping: *"Sales by category"*
    - Correlation: *"Correlation between age and salary"*
    - Time series: *"Trends over time"*
    - Top N: *"Top 10 customers"*
    """)

# File uploader
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    uploaded_files = st.file_uploader(
        "📁 Upload CSV files",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload one or more CSV files for analysis"
    )

# Initialize session state
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Load uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            # Try different encodings
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1')

            # Auto-detect datetime columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass

            st.session_state.dataframes[uploaded_file.name] = df
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {str(e)}")

# Main interface
if st.session_state.dataframes:

    # Overview metrics
    st.subheader("📊 Data Overview")

    total_rows = sum(df.shape[0] for df in st.session_state.dataframes.values())
    total_cols = sum(df.shape[1] for df in st.session_state.dataframes.values())
    total_files = len(st.session_state.dataframes)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📄 Files Uploaded", total_files)
    with col2:
        st.metric("📋 Total Rows", f"{total_rows:,}")
    with col3:
        st.metric("📊 Total Columns", total_cols)
    with col4:
        memory_usage = sum(df.memory_usage(deep=True).sum() for df in st.session_state.dataframes.values()) / 1024**2
        st.metric("💾 Memory", f"{memory_usage:.2f} MB")

    st.markdown("---")

    # Dataset selector
    if len(st.session_state.dataframes) > 1:
        selected_file = st.selectbox(
            "Select dataset for analysis:",
            list(st.session_state.dataframes.keys()),
            help="Choose which dataset to analyze"
        )
        df = st.session_state.dataframes[selected_file]
    else:
        selected_file = list(st.session_state.dataframes.keys())[0]
        df = st.session_state.dataframes[selected_file]

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data View", "📈 Statistics", "💬 Q&A Analysis", "📊 Visualizations"])

    with tab1:
        # Data preview with filters
        st.subheader(f"Data: {selected_file}")

        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("🔍 Search in data", "")
        with col2:
            rows_to_show = st.number_input("Rows to display", 10, 1000, 100)

        # Apply search filter
        display_df = df
        if search_term:
            mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            display_df = df[mask]
            st.info(f"Found {len(display_df)} rows matching '{search_term}'")

        st.dataframe(display_df.head(rows_to_show), use_container_width=True)

        # Download filtered data
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Data",
            csv,
            f"filtered_{selected_file}",
            "text/csv",
            key='download-csv'
        )

    with tab2:
        # Statistical summary
        st.subheader("📊 Statistical Summary")

        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.write("**Numeric Columns:**")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)

        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            st.write("**Categorical Columns:**")
            cat_summary = []
            for col in categorical_cols[:10]:  # Limit to 10
                cat_summary.append({
                    'Column': col,
                    'Unique Values': df[col].nunique(),
                    'Most Common': df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A',
                    'Missing': df[col].isna().sum()
                })
            st.dataframe(pd.DataFrame(cat_summary), use_container_width=True)

        # Correlation matrix for numeric columns
        if len(numeric_cols) > 1:
            st.write("**Correlation Matrix:**")
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix,
                           text_auto='.2f',
                           aspect="auto",
                           color_continuous_scale='RdBu_r',
                           title="Feature Correlations")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Advanced Q&A Interface
        st.subheader("💬 Ask Questions About Your Data")

        col1, col2 = st.columns([4, 1])
        with col1:
            question = st.text_input(
                "Enter your question:",
                placeholder="e.g., What is the average salary by department?"
            )
        with col2:
            st.write("")
            st.write("")
            analyze_btn = st.button("🔍 Analyze", type="primary")

        if analyze_btn and question.strip():

            def advanced_analyze(question, df):
                """
                Advanced question analysis with pattern matching and data operations
                """
                q_lower = question.lower().strip()

                try:
                    # BASIC COUNT QUERIES
                    if any(phrase in q_lower for phrase in ['how many', 'number of', 'count of', 'total rows']):
                        # Check for filtering conditions
                        for col in df.columns:
                            if col.lower() in q_lower:
                                # Filter by specific value
                                for val in df[col].dropna().astype(str).unique()[:100]:
                                    if val.lower() in q_lower:
                                        count = df[df[col].astype(str).str.lower() == val.lower()].shape[0]
                                        return {
                                            'answer': f"{count} rows",
                                            'details': f"Filtered by {col} = {val}",
                                            'type': 'count'
                                        }
                        return {'answer': f"{df.shape[0]} rows", 'details': 'Total row count', 'type': 'count'}

                    # AVERAGE/MEAN QUERIES
                    elif any(phrase in q_lower for phrase in ['average', 'mean', 'avg']):
                        for col in df.columns:
                            if col.lower() in q_lower and df[col].dtype in ['int64', 'float64']:
                                # Check for grouping
                                group_col = None
                                for gcol in df.columns:
                                    if gcol.lower() in q_lower and gcol != col and df[gcol].dtype == 'object':
                                        group_col = gcol
                                        break

                                if group_col:
                                    grouped = df.groupby(group_col)[col].mean().round(2)
                                    return {
                                        'answer': grouped.to_dict(),
                                        'details': f"Average {col} by {group_col}",
                                        'type': 'grouped_stat',
                                        'dataframe': grouped.reset_index()
                                    }
                                else:
                                    avg = df[col].mean()
                                    return {
                                        'answer': f"{avg:.2f}",
                                        'details': f"Average of {col}",
                                        'type': 'statistic'
                                    }

                    # SUM QUERIES
                    elif any(phrase in q_lower for phrase in ['total', 'sum']) and 'count' not in q_lower:
                        for col in df.columns:
                            if col.lower() in q_lower and df[col].dtype in ['int64', 'float64']:
                                # Check for grouping
                                group_col = None
                                for gcol in df.columns:
                                    if gcol.lower() in q_lower and gcol != col and df[gcol].dtype == 'object':
                                        group_col = gcol
                                        break

                                if group_col:
                                    grouped = df.groupby(group_col)[col].sum()
                                    return {
                                        'answer': grouped.to_dict(),
                                        'details': f"Total {col} by {group_col}",
                                        'type': 'grouped_stat',
                                        'dataframe': grouped.reset_index()
                                    }
                                else:
                                    total = df[col].sum()
                                    return {
                                        'answer': f"{total:,.2f}",
                                        'details': f"Sum of {col}",
                                        'type': 'statistic'
                                    }

                    # MAX/MIN QUERIES
                    elif any(phrase in q_lower for phrase in ['highest', 'maximum', 'max', 'largest', 'biggest', 'top']):
                        for col in df.columns:
                            if col.lower() in q_lower and df[col].dtype in ['int64', 'float64']:
                                # Check for TOP N
                                top_n = 5
                                if 'top' in q_lower:
                                    for word in q_lower.split():
                                        if word.isdigit():
                                            top_n = int(word)
                                            break

                                if 'top' in q_lower:
                                    top_values = df.nlargest(top_n, col)[[col]]
                                    return {
                                        'answer': f"Top {top_n} values",
                                        'details': f"Highest values in {col}",
                                        'type': 'top_n',
                                        'dataframe': top_values
                                    }
                                else:
                                    max_val = df[col].max()
                                    return {
                                        'answer': f"{max_val}",
                                        'details': f"Maximum value in {col}",
                                        'type': 'statistic'
                                    }

                    elif any(phrase in q_lower for phrase in ['lowest', 'minimum', 'min', 'smallest', 'bottom']):
                        for col in df.columns:
                            if col.lower() in q_lower and df[col].dtype in ['int64', 'float64']:
                                min_val = df[col].min()
                                return {
                                    'answer': f"{min_val}",
                                    'details': f"Minimum value in {col}",
                                    'type': 'statistic'
                                }

                    # UNIQUE VALUES
                    elif any(phrase in q_lower for phrase in ['unique', 'distinct', 'different']):
                        for col in df.columns:
                            if col.lower() in q_lower:
                                unique_vals = df[col].dropna().unique()
                                if len(unique_vals) <= 50:
                                    return {
                                        'answer': list(unique_vals),
                                        'details': f"Unique values in {col}",
                                        'type': 'list'
                                    }
                                else:
                                    return {
                                        'answer': f"{len(unique_vals)} unique values",
                                        'details': f"Too many to display all",
                                        'type': 'count'
                                    }

                    # CORRELATION QUERIES
                    elif 'correlation' in q_lower or 'correlated' in q_lower:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        mentioned_cols = [col for col in numeric_cols if col.lower() in q_lower]

                        if len(mentioned_cols) >= 2:
                            corr = df[mentioned_cols].corr().iloc[0, 1]
                            return {
                                'answer': f"{corr:.3f}",
                                'details': f"Correlation between {mentioned_cols[0]} and {mentioned_cols[1]}",
                                'type': 'correlation'
                            }

                    # MISSING VALUES
                    elif any(phrase in q_lower for phrase in ['missing', 'null', 'empty', 'blank']):
                        missing = df.isnull().sum()
                        missing = missing[missing > 0]
                        if len(missing) > 0:
                            return {
                                'answer': missing.to_dict(),
                                'details': 'Missing values per column',
                                'type': 'missing',
                                'dataframe': missing.reset_index()
                            }
                        else:
                            return {
                                'answer': 'No missing values',
                                'details': 'Dataset is complete',
                                'type': 'info'
                            }

                    # GROUPBY with COUNT
                    elif 'by' in q_lower and any(phrase in q_lower for phrase in ['group', 'breakdown', 'distribution']):
                        # Find grouping column
                        for col in df.columns:
                            if col.lower() in q_lower and df[col].dtype == 'object':
                                grouped = df[col].value_counts().head(20)
                                return {
                                    'answer': grouped.to_dict(),
                                    'details': f"Distribution of {col}",
                                    'type': 'distribution',
                                    'dataframe': grouped.reset_index()
                                }

                    # PERCENTAGE/PROPORTION
                    elif any(phrase in q_lower for phrase in ['percentage', 'percent', 'proportion']):
                        for col in df.columns:
                            if col.lower() in q_lower:
                                if df[col].dtype == 'object':
                                    value_counts = df[col].value_counts()
                                    percentages = (value_counts / len(df) * 100).round(2)
                                    return {
                                        'answer': percentages.to_dict(),
                                        'details': f"Percentage distribution of {col}",
                                        'type': 'percentage',
                                        'dataframe': percentages.reset_index()
                                    }

                    # TREND/TIME SERIES
                    elif any(phrase in q_lower for phrase in ['trend', 'over time', 'timeline', 'time series']):
                        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                        if datetime_cols:
                            date_col = datetime_cols[0]
                            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                            if numeric_cols:
                                value_col = numeric_cols[0]
                                trend_data = df.groupby(df[date_col].dt.to_period('M'))[value_col].mean()
                                return {
                                    'answer': 'Trend data available',
                                    'details': f"Trend of {value_col} over {date_col}",
                                    'type': 'trend',
                                    'dataframe': trend_data.reset_index()
                                }

                    return None

                except Exception as e:
                    return None

            result = advanced_analyze(question, df)

            if result:
                # Add to history
                st.session_state.analysis_history.append({
                    'question': question,
                    'result': result,
                    'timestamp': datetime.now()
                })

                # Display result
                st.success("✅ Analysis Complete")

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**Question:** {question}")
                    st.markdown(f"**Type:** {result['type'].replace('_', ' ').title()}")
                    st.markdown(f"**Details:** {result['details']}")

                # Display answer based on type
                if result['type'] in ['grouped_stat', 'distribution', 'percentage', 'missing', 'top_n', 'trend']:
                    if 'dataframe' in result:
                        st.dataframe(result['dataframe'], use_container_width=True)

                        # Auto-generate chart
                        if result['type'] in ['distribution', 'grouped_stat', 'percentage']:
                            fig = px.bar(
                                result['dataframe'],
                                x=result['dataframe'].columns[0],
                                y=result['dataframe'].columns[1],
                                title=result['details']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        elif result['type'] == 'trend':
                            fig = px.line(
                                result['dataframe'],
                                x=result['dataframe'].columns[0],
                                y=result['dataframe'].columns[1],
                                title=result['details']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                elif result['type'] == 'list':
                    st.write(result['answer'])
                else:
                    st.markdown(f"### 📊 Answer: **{result['answer']}**")

            else:
                st.error("❌ Could not answer this question")
                st.info("""
                💡 **Tips:**
                - Make sure column names are mentioned in your question
                - Try simpler phrasings
                - Check the supported query types in the sidebar
                """)

    with tab4:
        # Visualizations
        st.subheader("📊 Data Visualizations")

        viz_type = st.selectbox(
            "Select visualization type:",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart", "Heatmap"]
        )

        col1, col2 = st.columns(2)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        all_cols = df.columns.tolist()

        with col1:
            x_col = st.selectbox("X-axis", all_cols)
        with col2:
            y_col = st.selectbox("Y-axis", all_cols if viz_type != "Pie Chart" else numeric_cols)

        if st.button("Generate Visualization"):
            try:
                if viz_type == "Bar Chart":
                    fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                elif viz_type == "Line Chart":
                    fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                elif viz_type == "Scatter Plot":
                    fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                elif viz_type == "Histogram":
                    fig = px.histogram(df, x=x_col, title=f"Distribution of {x_col}")
                elif viz_type == "Box Plot":
                    fig = px.box(df, x=x_col, y=y_col, title=f"{y_col} distribution by {x_col}")
                elif viz_type == "Pie Chart":
                    value_counts = df[x_col].value_counts().head(10)
                    fig = px.pie(values=value_counts.values, names=value_counts.index, title=f"Distribution of {x_col}")
                elif viz_type == "Heatmap":
                    if len(numeric_cols) > 1:
                        corr = df[numeric_cols].corr()
                        fig = px.imshow(corr, text_auto='.2f', title="Correlation Heatmap")

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")

    # Analysis History
    if st.session_state.analysis_history:
        with st.expander("📜 Analysis History"):
            for i, item in enumerate(reversed(st.session_state.analysis_history[-10:])):
                st.markdown(f"**{i+1}.** {item['question']} - *{item['timestamp'].strftime('%H:%M:%S')}*")

else:
    # Welcome screen
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h2>👋 Welcome to Advanced CSV Data Analyst</h2>
        <p style='font-size: 1.2rem; color: #666;'>
            Upload your CSV files to get started with powerful data analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🔍 Smart Analysis
        Ask questions in natural language and get instant answers
        """)

    with col2:
        st.markdown("""
        ### 📊 Rich Visualizations
        Create beautiful charts and graphs automatically
        """)

    with col3:
        st.markdown("""
        ### 📈 Statistical Insights
        Get comprehensive statistical summaries
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Made with ❤️ using Streamlit | Advanced CSV Data Analyst v2.0
</div>
""", unsafe_allow_html=True)
