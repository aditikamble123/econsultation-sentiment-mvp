"""
E-consultation Sentiment Analysis MVP
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime
import tempfile
import plotly.graph_objs as go
import plotly.express as px
from io import BytesIO
import base64

# Download required NLTK data at startup
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
for corpus in ['punkt', 'brown', 'stopwords', 'averaged_perceptron_tagger', 'punkt_tab']:
    try:
        nltk.download(corpus, quiet=True)
    except:
        pass

# Import our modules
from sentiment_analyzer import SentimentAnalyzer
from text_summarizer import TextSummarizer
from wordcloud_generator import WordCloudGenerator

# Page configuration
st.set_page_config(
    page_title="E-consultation Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize analyzers
@st.cache_resource
def load_analyzers():
    """Load and cache the analysis models"""
    sentiment_analyzer = SentimentAnalyzer()
    text_summarizer = TextSummarizer()
    wordcloud_generator = WordCloudGenerator()
    return sentiment_analyzer, text_summarizer, wordcloud_generator

sentiment_analyzer, text_summarizer, wordcloud_generator = load_analyzers()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: transparent;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def perform_analysis(df):
    """Main analysis function"""
    results = {}
    
    with st.spinner("Performing sentiment analysis..."):
        # 1. Sentiment Analysis
        sentiment_results = sentiment_analyzer.analyze(df)
        results['sentiment_summary'] = sentiment_results['summary']
        results['detailed'] = sentiment_results['detailed']  # Fixed: using consistent key
        results['sentiment_by_provision'] = sentiment_results['by_provision']
        results['sentiment_by_stakeholder'] = sentiment_results['by_stakeholder']
    
    with st.spinner("Generating text summaries..."):
        # 2. Text Summarization
        summary_results = text_summarizer.summarize(df, sentiment_results['detailed'])
        results['overall_summary'] = summary_results['overall_summary']
        results['key_themes'] = summary_results['key_themes']
        results['provision_summaries'] = summary_results['provision_summaries']
    
    with st.spinner("Creating word clouds..."):
        # 3. Word Cloud Generation
        wordcloud_results = wordcloud_generator.generate(df, sentiment_results['detailed'])
        results['main_wordcloud'] = wordcloud_results['main_wordcloud']
        results['sentiment_wordclouds'] = wordcloud_results['sentiment_wordclouds']
    
    with st.spinner("Creating visualizations..."):
        # 4. Create charts
        results['charts'] = create_charts(results)
    
    # 5. Statistical summary
    results['statistics'] = {
        'total_comments': len(df),
        'unique_stakeholders': df['stakeholder_name'].nunique(),
        'provisions_discussed': df['provision_reference'].nunique(),
        'avg_comment_length': df['comment_text'].str.len().mean()
    }
    
    return results

def create_charts(results):
    """Create Plotly charts for visualization"""
    charts = {}
    
    # 1. Overall sentiment distribution
    sentiment_data = results['sentiment_summary']
    fig_pie = go.Figure(data=[go.Pie(
        labels=list(sentiment_data.keys()),
        values=list(sentiment_data.values()),
        hole=0.3,
        marker_colors=['#10b981', '#ef4444', '#6b7280']
    )])
    fig_pie.update_layout(
        title="Overall Sentiment Distribution",
        height=400
    )
    charts['sentiment_pie'] = fig_pie
    
    # 2. Sentiment by provision
    provision_df = pd.DataFrame(results['sentiment_by_provision']).T
    if not provision_df.empty:
        fig_bar = go.Figure()
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in provision_df.columns:
                fig_bar.add_trace(go.Bar(
                    name=sentiment.capitalize(),
                    x=provision_df.index,
                    y=provision_df[sentiment],
                    marker_color={'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#6b7280'}[sentiment]
                ))
        fig_bar.update_layout(
            title="Sentiment Distribution by Provision",
            xaxis_title="Provision",
            yaxis_title="Number of Comments",
            barmode='stack',
            height=400
        )
        charts['provision_bar'] = fig_bar
    
    # 3. Top stakeholders
    stakeholder_df = pd.DataFrame(results['sentiment_by_stakeholder']).T
    if not stakeholder_df.empty:
        stakeholder_df['total'] = stakeholder_df.sum(axis=1)
        top_stakeholders = stakeholder_df.nlargest(10, 'total')
        
        fig_stakeholder = go.Figure()
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in top_stakeholders.columns and sentiment != 'total':
                fig_stakeholder.add_trace(go.Bar(
                    name=sentiment.capitalize(),
                    x=top_stakeholders.index,
                    y=top_stakeholders[sentiment],
                    marker_color={'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#6b7280'}[sentiment]
                ))
        fig_stakeholder.update_layout(
            title="Top 10 Most Active Stakeholders",
            xaxis_title="Stakeholder",
            yaxis_title="Number of Comments",
            barmode='stack',
            height=400,
            xaxis_tickangle=-45
        )
        charts['stakeholder_bar'] = fig_stakeholder
    
    return charts

def display_results(results):
    """Display analysis results in Streamlit"""
    
    # Header metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Comments", results['statistics']['total_comments'])
    with col2:
        st.metric("Unique Stakeholders", results['statistics']['unique_stakeholders'])
    with col3:
        st.metric("Provisions Discussed", results['statistics']['provisions_discussed'])
    with col4:
        st.metric("Avg Comment Length", f"{results['statistics']['avg_comment_length']:.0f} chars")
    
    st.divider()
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "💭 Sentiment Analysis", "📝 Summaries", "☁️ Word Clouds", "📈 Detailed Data"])
    
    with tab1:
        st.header("Analysis Overview")
        
        # Overall summary
        st.subheader("Executive Summary")
        st.info(results['overall_summary'])
        
        # Key themes
        st.subheader("Key Themes Identified")
        for i, theme in enumerate(results['key_themes'], 1):
            st.write(f"{i}. {theme}")
        
        # Overall sentiment chart
        st.plotly_chart(results['charts']['sentiment_pie'], use_container_width=True)
    
    with tab2:
        st.header("Sentiment Analysis Results")
        
        # Sentiment distribution
        col1, col2, col3 = st.columns(3)
        sentiment_summary = results['sentiment_summary']
        total = sum(sentiment_summary.values())
        
        with col1:
            positive_pct = (sentiment_summary.get('positive', 0) / total * 100) if total > 0 else 0
            st.metric("Positive", f"{sentiment_summary.get('positive', 0)} ({positive_pct:.1f}%)")
        with col2:
            negative_pct = (sentiment_summary.get('negative', 0) / total * 100) if total > 0 else 0
            st.metric("Negative", f"{sentiment_summary.get('negative', 0)} ({negative_pct:.1f}%)")
        with col3:
            neutral_pct = (sentiment_summary.get('neutral', 0) / total * 100) if total > 0 else 0
            st.metric("Neutral", f"{sentiment_summary.get('neutral', 0)} ({neutral_pct:.1f}%)")
        
        # Charts
        # Removed provision bar chart as requested
        
        if 'stakeholder_bar' in results['charts']:
            st.plotly_chart(results['charts']['stakeholder_bar'], use_container_width=True)
    
    with tab3:
        st.header("Text Summaries")
        
        # Provision summaries
        st.subheader("Summaries by Provision")
        for provision, summary in results['provision_summaries'].items():
            with st.expander(f"📄 {provision}"):
                st.write(summary)
    
    with tab4:
        st.header("Word Cloud Visualizations")
        
        # Main word cloud
        st.subheader("Overall Word Cloud")
        if results['main_wordcloud']:
            st.image(f"data:image/png;base64,{results['main_wordcloud']}", use_column_width=True)
        
        # Sentiment word clouds
        st.subheader("Word Clouds by Sentiment")
        if results['sentiment_wordclouds']:
            cols = st.columns(3)
            sentiments = ['positive', 'negative', 'neutral']
            for i, sentiment in enumerate(sentiments):
                if sentiment in results['sentiment_wordclouds']:
                    with cols[i]:
                        st.write(f"**{sentiment.capitalize()}**")
                        st.image(f"data:image/png;base64,{results['sentiment_wordclouds'][sentiment]}", use_column_width=True)
    
    with tab5:
        st.header("Detailed Data Tables")
        
        # Detailed sentiments table
        st.subheader("Comment-level Sentiment Analysis")
        detailed_df = results['detailed']  # Fixed: using correct key
        st.dataframe(detailed_df, use_container_width=True)
        
        # Download options
        csv = detailed_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Detailed Results as CSV",
            data=csv,
            file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header"><h1>📊 E-consultation Sentiment Analysis</h1><p>Analyze stakeholder feedback with AI-powered insights</p></div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        This tool analyzes e-consultation comments to provide:
        - Sentiment analysis (positive/negative/neutral)
        - Key themes and insights
        - Stakeholder engagement metrics
        - Visual representations of feedback
        """)
        
        st.header("Instructions")
        st.markdown("""
        1. Upload a CSV or Excel file with columns:
           - `comment_id`
           - `stakeholder_name`
           - `comment_text`
           - `provision_reference`
        2. Click 'Analyze' to process
        3. Explore results in different tabs
        """)
        
        # Sample data option
        if st.button("Load Sample Data"):
            try:
                # Load the sample_data.csv file
                sample_data = pd.read_csv('sample_data.csv')
                st.session_state['uploaded_data'] = sample_data
                st.success(f"Sample data loaded! ({len(sample_data)} comments)")
                st.rerun()
            except FileNotFoundError:
                # Fallback to hardcoded sample if file not found
                sample_data = pd.DataFrame({
                    'comment_id': [1, 2, 3, 4, 5],
                    'stakeholder_name': ['John Doe', 'Jane Smith', 'ACME Corp', 'John Doe', 'Green Initiative'],
                    'comment_text': [
                        'This provision will greatly improve transparency in the sector.',
                        'Concerned about the implementation timeline being too aggressive.',
                        'We support this initiative as it aligns with industry best practices.',
                        'The cost implications need further consideration.',
                        'Excellent proposal for environmental sustainability.'
                    ],
                    'provision_reference': ['Section 2.1', 'Section 3.2', 'Section 2.1', 'Section 3.2', 'Section 4.1']
                })
                st.session_state['uploaded_data'] = sample_data
                st.success("Sample data loaded!")
                st.rerun()
    
    # Main content area
    if 'uploaded_data' not in st.session_state:
        # File upload section
        st.header("Upload Your Data")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="File must contain: comment_id, stakeholder_name, comment_text, provision_reference"
        )
        
        if uploaded_file is not None:
            try:
                # Read the file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Validate columns
                required_columns = ['comment_id', 'stakeholder_name', 'comment_text', 'provision_reference']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    st.stop()
                
                # Clean data
                df = df.dropna(subset=['comment_text'])
                
                if len(df) == 0:
                    st.error("No valid comments found in the file")
                    st.stop()
                
                # Show preview
                st.subheader("Data Preview")
                st.write(f"Found {len(df)} valid comments")
                st.dataframe(df.head(), use_container_width=True)
                
                # Analyze button
                if st.button("🚀 Analyze Comments", type="primary", use_container_width=True):
                    st.session_state['uploaded_data'] = df
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    else:
        # Analysis section
        df = st.session_state['uploaded_data']
        
        # Add option to upload new file
        if st.button("📁 Upload New File"):
            del st.session_state['uploaded_data']
            if 'analysis_results' in st.session_state:
                del st.session_state['analysis_results']
            st.rerun()
        
        # Perform or retrieve analysis
        if 'analysis_results' not in st.session_state:
            with st.spinner("Analyzing your data... This may take a few moments."):
                results = perform_analysis(df)
                st.session_state['analysis_results'] = results
        
        # Display results
        display_results(st.session_state['analysis_results'])

if __name__ == "__main__":
    main()
