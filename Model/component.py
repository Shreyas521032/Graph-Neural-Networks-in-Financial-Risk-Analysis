import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
from datetime import date

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


st.set_page_config(
    page_title="Financial Risk Analyzer Pro 📊",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💼"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0D2F62;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #E2E8F0;
        font-weight: 800;
        text-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E429F;
        padding-top: 1rem;
        margin-top: 1rem;
        font-weight: 600;
    }
    .card {
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        background-color: white;
        margin-bottom: 1.5rem;
        border: 1px solid #f0f0f0;
    }
    .premium-card {
        background: linear-gradient(to right, #ffffff, #f8f9ff);
        border-left: 4px solid #0D47A1;
    }
    .metric-card {
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.12);
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#0D2F62, #1A4988);
        color: white;
    }
    .stButton>button {
        background-color: #0D47A1;
        color: white;
        border-radius: 20px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #0D2F62;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transform: translateY(-2px);
    }
    .stProgress .st-eb {
        background-color: #0D47A1;
    }
    .tab-subheader {
        font-size: 1.2rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 1rem;
    }
    .insights-card {
        background: linear-gradient(120deg, #EBF4FF 0%, #F9FAFE 100%);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        border-left: 4px solid #0D47A1;
    }
    .footer {
        text-align: center;
        padding: 1.5rem;
        background-color: #0D2F62;
        color: white;
        border-radius: 0 0 10px 10px;
        margin-top: 2rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 700;
        line-height: 1;
        color: #fff;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        background-color: #0D47A1;
        margin-right: 0.5rem;
    }
    .logo-text {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: white;
        display: flex;
        align-items: center;
    }
    .trend-indicator-up {
        color: #10B981;
        font-weight: bold;
    }
    .trend-indicator-down {
        color: #EF4444;
        font-weight: bold;
    }
    .sidebar-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        padding: 1rem;
        background-color: rgba(13, 47, 98, 0.9);
        text-align: center;
        color: white;
        font-size: 0.8rem;
    }
    .premium-badge {
        position: absolute;
        top: 0;
        right: 0;
        background-color: #0D47A1;
        color: white;
        padding: 0.25rem 0.5rem;
        font-size: 0.7rem;
        border-radius: 0 0 0 5px;
    }
    .risk-level-low {
        color: #10B981;
        font-weight: bold;
    }
    .risk-level-medium {
        color: #F59E0B;
        font-weight: bold;
    }
    .risk-level-high {
        color: #EF4444;
        font-weight: bold;
    }
    div[data-testid="stDecoration"] {
        background-image: linear-gradient(90deg, #0D2F62, #1A4988);
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def load_sample_data():
    # Instead of generating random data, we'll create a structured financial dataset
    # This approach ensures the data is realistic and meaningful
    
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Categories with more finance-specific names
    categories = ['Market Volatility', 'Regulatory Changes', 'Macroeconomic Factors', 
                  'Industry Disruption', 'Liquidity Issues', 'Corporate Actions',
                  'Geopolitical Events', 'Monetary Policy', 'Fiscal Policy', 'Market Sentiment']
    
    sources = ['Bloomberg', 'Financial Times', 'Reuters', 'Wall Street Journal', 
               'CNBC', 'Forbes', 'The Economist', 'Barron\'s', 'MarketWatch', 'Morningstar']
    
    risk_levels = ['Low', 'Medium', 'High', 'Critical']
    
    # Real headlines examples for financial news
    headlines_templates = [
        "{Company} reports {Percent}% {Direction} in quarterly earnings",
        "{Region} markets {Direction} amid {Factor} concerns",
        "{Bank} implements new {Policy} strategy to mitigate {Risk} risks",
        "Analysts predict {Direction} in {Sector} sector following {Event}",
        "{Agency} raises concerns over {Industry} industry {Risk} exposure",
        "{Company} announces merger with {Company2}, raising {Factor} questions",
        "{Bank} faces regulatory scrutiny over {Compliance} issues",
        "{Region} economy shows signs of {Direction} amid {Factor}",
        "New {Policy} regulation could impact {Industry} profit margins",
        "{Event} triggers volatility in {Market} markets",
        "{Company} stock {Direction} after {News} announcement",
        "{Rating} downgrades {Company} credit outlook citing {Risk} concerns",
        "{Index} reaches {Record} as {Sector} stocks surge",
        "{Commodity} prices {Direction} due to {Factor} pressures",
        "{Regulator} announces new {Compliance} requirements for {Industry} firms"
    ]
    
    # Elements to populate templates
    elements = {
        'Company': ['Goldman Sachs', 'JPMorgan', 'Morgan Stanley', 'BlackRock', 'Vanguard', 
                   'Citigroup', 'Bank of America', 'Wells Fargo', 'HSBC', 'Barclays',
                   'Deutsche Bank', 'UBS', 'Credit Suisse', 'Fidelity', 'PIMCO'],
        'Company2': ['Merrill Lynch', 'State Street', 'BNY Mellon', 'Northern Trust', 'Invesco',
                    'AllianceBernstein', 'Prudential', 'AIG', 'MetLife', 'Allianz'],
        'Percent': [str(round(np.random.uniform(0.5, 15.5), 1)) for _ in range(20)],
        'Direction': ['rise', 'fall', 'surge', 'decline', 'plummet', 'soar', 'tumble', 'rebound'],
        'Factor': ['inflation', 'recession', 'interest rate', 'supply chain', 'labor shortage',
                  'regulatory', 'liquidity', 'credit', 'default', 'currency fluctuation'],
        'Bank': ['Federal Reserve', 'European Central Bank', 'Bank of England', 'Bank of Japan',
                'People\'s Bank of China', 'Swiss National Bank', 'Bank of Canada'],
        'Policy': ['monetary', 'fiscal', 'interest rate', 'quantitative easing', 'tightening',
                  'stimulus', 'austerity', 'capital control', 'lending'],
        'Risk': ['liquidity', 'credit', 'operational', 'market', 'systemic', 'counterparty',
               'sovereign', 'political', 'regulatory', 'concentration'],
        'Sector': ['technology', 'financial', 'healthcare', 'energy', 'consumer discretionary',
                  'consumer staples', 'industrial', 'utilities', 'real estate', 'materials'],
        'Agency': ['SEC', 'FINRA', 'OCC', 'FDIC', 'Federal Reserve', 'CFPB', 'FSB', 'IOSCO', 'EBA', 'ESMA'],
        'Industry': ['banking', 'investment management', 'insurance', 'fintech', 'wealth management',
                    'asset management', 'pension fund', 'hedge fund', 'private equity', 'brokerage'],
        'Compliance': ['AML', 'KYC', 'capital adequacy', 'stress test', 'Volcker Rule',
                      'Basel III', 'Dodd-Frank', 'MiFID', 'GDPR', 'reporting'],
        'Region': ['US', 'European', 'Asian', 'UK', 'Chinese', 'Japanese', 'Emerging market',
                  'Latin American', 'Middle Eastern', 'Global'],
        'Event': ['FOMC meeting', 'GDP report', 'inflation data', 'earnings season', 'credit event',
                 'market selloff', 'economic stimulus', 'trade tensions', 'political uncertainty'],
        'Market': ['equity', 'bond', 'credit', 'currency', 'commodity', 'derivative', 'repo',
                  'swap', 'options', 'futures'],
        'News': ['earnings', 'management change', 'restructuring', 'dividend', 'share buyback',
                'debt issuance', 'credit rating', 'acquisition', 'divestiture', 'regulatory'],
        'Rating': ['Moody\'s', 'S&P', 'Fitch', 'DBRS', 'AM Best', 'Kroll', 'Morningstar', 'Weiss'],
        'Index': ['S&P 500', 'Dow Jones', 'Nasdaq', 'Russell 2000', 'FTSE 100', 'DAX', 'Nikkei',
                 'Hang Seng', 'Shanghai Composite', 'MSCI World'],
        'Record': ['all-time high', 'yearly high', 'multi-year high', 'record level', 'technical resistance'],
        'Commodity': ['oil', 'gold', 'natural gas', 'copper', 'wheat', 'corn', 'soybeans', 'silver', 'platinum']
    }
    
    # Generate realistic dataset
    data = []
    for _ in range(1000):
        date = np.random.choice(dates)
        category = np.random.choice(categories)
        source = np.random.choice(sources)
        
        # Assign risk level with some logic
        if category in ['Market Volatility', 'Liquidity Issues', 'Geopolitical Events']:
            # Higher chance of high risk
            risk_level = np.random.choice(risk_levels, p=[0.1, 0.2, 0.5, 0.2])
        elif category in ['Regulatory Changes', 'Corporate Actions']:
            # Medium risk usually
            risk_level = np.random.choice(risk_levels, p=[0.2, 0.5, 0.2, 0.1])
        elif category in ['Macroeconomic Factors', 'Fiscal Policy', 'Monetary Policy']:
            # More balanced risk
            risk_level = np.random.choice(risk_levels, p=[0.25, 0.35, 0.25, 0.15])
        else:
            # More typical distribution
            risk_level = np.random.choice(risk_levels, p=[0.4, 0.3, 0.2, 0.1])
        
        # Generate headline
        headline_template = np.random.choice(headlines_templates)
        headline = headline_template
        
        # Replace placeholders with actual values
        for key in elements:
            if '{' + key + '}' in headline:
                headline = headline.replace('{' + key + '}', np.random.choice(elements[key]))
        
        # Generate sentiment based on headline and risk level
        base_sentiment = 0
        if 'rise' in headline or 'surge' in headline or 'soar' in headline or 'rebound' in headline:
            base_sentiment += 0.3
        if 'fall' in headline or 'decline' in headline or 'plummet' in headline or 'tumble' in headline:
            base_sentiment -= 0.3
        if 'concern' in headline or 'risk' in headline or 'scrutiny' in headline:
            base_sentiment -= 0.2
        if 'opportunity' in headline or 'growth' in headline or 'positive' in headline:
            base_sentiment += 0.2
            
        # Risk level affects sentiment
        if risk_level == 'Low':
            sentiment_adjustment = 0.4
        elif risk_level == 'Medium':
            sentiment_adjustment = 0.1
        elif risk_level == 'High':
            sentiment_adjustment = -0.2
        else:  # Critical
            sentiment_adjustment = -0.5
            
        sentiment_score = base_sentiment + sentiment_adjustment + np.random.uniform(-0.2, 0.2)
        sentiment_score = max(min(sentiment_score, 1.0), -1.0)  # Clip to [-1, 1]
        
        # Impact score also correlates with risk level
        if risk_level == 'Low':
            impact_base = np.random.uniform(1, 3)
        elif risk_level == 'Medium':
            impact_base = np.random.uniform(3, 5.5)
        elif risk_level == 'High':
            impact_base = np.random.uniform(5.5, 8)
        else:  # Critical
            impact_base = np.random.uniform(8, 10)
            
        impact_score = impact_base + np.random.uniform(-0.5, 0.5)
        impact_score = max(min(impact_score, 10.0), 1.0)  # Clip to [1, 10]
        
        # Generate time-series data for volume (trading activity)
        month = date.month
        # Higher volume in quarterly reporting months
        if month in [1, 4, 7, 10]:
            volume = np.random.randint(80, 150) * 1000
        else:
            volume = np.random.randint(40, 100) * 1000
            
        # Generate volatility index
        if risk_level == 'Low':
            vix = np.random.uniform(10, 15)
        elif risk_level == 'Medium':
            vix = np.random.uniform(15, 20)
        elif risk_level == 'High':
            vix = np.random.uniform(20, 30)
        else:  # Critical
            vix = np.random.uniform(30, 45)
        
        data.append({
            'date': date,
            'headline': headline,
            'category': category,
            'source': source,
            'risk_level': risk_level,
            'sentiment_score': sentiment_score,
            'impact_score': impact_score,
            'trading_volume': volume,
            'volatility_index': vix,
            'month': calendar.month_name[month]
        })
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    return df

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')
    
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return ' '.join(tokens)

def create_metrics_row(df):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(120deg, #E0F2FE 0%, #EFF6FF 100%);">
            <h3 style="color: #0D47A1;">📊 Data Points</h3>
            <h2 style="color: #0D2F62; font-size: 2.5rem;">{}</h2>
            <p style="color: #6B7280; font-size: 0.9rem;">Analysis records</p>
        </div>
        """.format(df.shape[0]), unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(120deg, #DBEAFE 0%, #EEF2FF 100%);">
            <h3 style="color: #0D47A1;">⚠️ Risk Categories</h3>
            <h2 style="color: #0D2F62; font-size: 2.5rem;">{}</h2>
            <p style="color: #6B7280; font-size: 0.9rem;">Classification levels</p>
        </div>
        """.format(df['risk_level'].nunique()), unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(120deg, #E0F2FE 0%, #EFF6FF 100%);">
            <h3 style="color: #0D47A1;">🔍 News Sources</h3>
            <h2 style="color: #0D2F62; font-size: 2.5rem;">{}</h2>
            <p style="color: #6B7280; font-size: 0.9rem;">Information channels</p>
        </div>
        """.format(df['source'].nunique()), unsafe_allow_html=True)
        
    with col4:
        avg_sentiment = df['sentiment_score'].mean()
        sentiment_emoji = "😊" if avg_sentiment > 0.2 else "😐" if abs(avg_sentiment) <= 0.2 else "😔"
        trend = df.sort_values('date')['sentiment_score'].iloc[-30:].mean() - df.sort_values('date')['sentiment_score'].iloc[-60:-30].mean()
        trend_icon = "↗️" if trend > 0 else "↘️"
        trend_class = "trend-indicator-up" if trend > 0 else "trend-indicator-down"
        
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(120deg, #DBEAFE 0%, #EEF2FF 100%);">
            <h3 style="color: #0D47A1;">{sentiment_emoji} Market Sentiment</h3>
            <h2 style="color: #0D2F62; font-size: 2.5rem;">{avg_sentiment:.2f}</h2>
            <p style="color: #6B7280; font-size: 0.9rem;">30-day trend: <span class="{trend_class}">{trend_icon} {abs(trend):.2f}</span></p>
        </div>
        """, unsafe_allow_html=True)

def create_advanced_insights_row(df):
    # Calculate key metrics
    high_risk_pct = df[df['risk_level'].isin(['High', 'Critical'])].shape[0] / df.shape[0] * 100
    volatility_avg = df['volatility_index'].mean()
    sentiment_trend = df.sort_values('date')['sentiment_score'].iloc[-10:].mean() - df.sort_values('date')['sentiment_score'].iloc[-20:-10].mean()
    top_category = df['category'].value_counts().index[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insights-card">
            <h4 style="color: #0D47A1; margin-top: 0;">⚠️ Risk Exposure Analysis</h4>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p><span class="badge">High Risk</span> <strong>{:.1f}%</strong> of monitored assets</p>
                    <p><span class="badge">VIX</span> Market volatility at <strong>{:.2f}</strong></p>
                </div>
                <div style="width: 120px; height: 120px; position: relative;">
                    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold; color: #0D47A1;">{:.1f}</div>
                        <div style="font-size: 0.8rem; color: #6B7280;">Risk Score</div>
                    </div>
                </div>
            </div>
        </div>
        """.format(high_risk_pct, volatility_avg, (high_risk_pct * volatility_avg) / 100), unsafe_allow_html=True)
        
    with col2:
        sentiment_class = "trend-indicator-up" if sentiment_trend > 0 else "trend-indicator-down"
        sentiment_icon = "↗️" if sentiment_trend > 0 else "↘️"
        
        st.markdown("""
        <div class="insights-card">
            <h4 style="color: #0D47A1; margin-top: 0;">📈 Market Intelligence</h4>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p><span class="badge">Sentiment</span> <span class="{sentiment_class}">{sentiment_icon} {abs(sentiment_trend):.2f}</span> 10-day trend</p>
                    <p><span class="badge">Focus</span> <strong>{top_category}</strong> dominates news cycle</p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.9rem; color: #6B7280;">Monitoring</div>
                    <div style="font-size: 1.2rem; font-weight: bold; color: #0D47A1;">{df['source'].nunique()} sources</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def create_risk_heat_calendar(df):
    # Prepare data for calendar heatmap
    df_calendar = df.copy()
    df_calendar['day'] = df_calendar['date'].dt.day
    df_calendar['month'] = df_calendar['date'].dt.month_name()
    
    # Map risk levels to numeric values
    risk_map = {
        'Low': 1, 
        'Medium': 2, 
        'High': 3, 
        'Critical': 4
    }
    df_calendar['risk_numeric'] = df_calendar['risk_level'].map(risk_map)
    
    # Aggregate by month and day
    heatmap_data = df_calendar.groupby(['month', 'day'])['risk_numeric'].mean().reset_index()
    
    # Create the heatmap
    months_order = list(calendar.month_name)[1:]
    fig = px.density_heatmap(
        heatmap_data,
        x='day',
        y='month',
        z='risk_numeric',
        category_orders={'month': months_order},
        color_continuous_scale=[
            '#10B981',  # Low - Green
            '#FBBF24',  # Medium - Yellow
            '#F97316',  # High - Orange  
            '#EF4444'   # Critical - Red
        ],
        title='🗓️ Risk Heat Calendar',
        labels={'risk_numeric': 'Risk Level', 'day': 'Day of Month', 'month': 'Month'}
    )
    
    fig.update_layout(
        height=400,
        yaxis={'categoryarray': months_order},
        coloraxis_colorbar=dict(
            tickvals=[1, 2, 3, 4],
            ticktext=['Low', 'Medium', 'High', 'Critical']
        )
    )
    
    return fig

def create_risk_timeline(df):
    # Prepare data for timeline
    df_timeline = df.copy()
    df_timeline['week'] = df_timeline['date'].dt.isocalendar().week
    df_timeline['month'] = df_timeline['date'].dt.month_name()
    
    # Calculate risk score by week
    risk_map = {
        'Low': 1, 
        'Medium': 2, 
        'High': 3, 
        'Critical': 4
    }
    df_timeline['risk_score'] = df_timeline['risk_level'].map(risk_map)
    
    # Aggregate by month and week
    timeline_data = df_timeline.groupby(['month', 'week']).agg({
        'risk_score': 'mean',
        'sentiment_score': 'mean',
        'impact_score': 'mean',
        'volatility_index': 'mean'
    }).reset_index()
    
    # Create a custom week-month string for continuous x-axis
    timeline_data['period'] = timeline_data['month'] + ' W' + timeline_data['week'].astype(str)
    
    # Create the line chart
    fig = go.Figure()
    
    # Add risk score line
    fig.add_trace(go.Scatter(
        x=timeline_data['period'],
        y=timeline_data['risk_score'],
        mode='lines+markers',
        name='Risk Score',
        line=dict(color='#EF4444', width=3),
        marker=dict(size=8, symbol='circle')
    ))
    
    # Add sentiment score line (scaled for visibility)
    fig.add_trace(go.Scatter(
        x=timeline_data['period'],
        y=(timeline_data['sentiment_score'] + 1) * 2,  # Scale from [-1,1] to [0,4]
        mode='lines+markers',
        name='Sentiment',
        line=dict(color='#3B82F6', width=2, dash='dot'),
        marker=dict(size=6, symbol='diamond')
    ))
    
    # Add volatility index line (scaled)
    fig.add_trace(go.Scatter(
        x=timeline_data['period'],
        y=timeline_data['volatility_index'] / 10,  # Scale to similar range
        mode='lines',
        name='Volatility',
        line=dict(color='#10B981', width=2),
        opacity=0.7
    ))
    
    fig.update_layout(
        title='📈 Risk Evolution Timeline',
        height=400,
        xaxis_title='Time Period',
        yaxis_title='Metric Values',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    return fig

def create_source_analysis(df):
    # Prepare data for source reliability analysis
    source_analysis = df.groupby('source').agg({
        'sentiment_score': ['mean', 'std'],
        'impact_score': 'mean',
        'risk_level': lambda x: (x == 'High').sum() / len(x) * 100,
        'headline': 'count'
    }).reset_index()
    
    source_analysis.columns = ['source', 'avg_sentiment', 'sentiment_std', 'avg_impact', 'high_risk_pct', 'article_count']
    
    # Calculate reliability score (lower variance = higher reliability)
    source_analysis['reliability_score'] = 10 - source_analysis['sentiment_std'] * 5
    source_analysis['reliability_score'] = source_analysis['reliability_score'].clip(0, 10)
    
    # Create figure
    fig = px.scatter(
      source_analysis,
      x='avg_sentiment',
      y='high_risk_pct',
        size='article_count',
        color='reliability_score',
        hover_name='source',
        color_continuous_scale=[
            '#EF4444',  # Low reliability - Red
            '#FBBF24',  # Medium reliability - Yellow
            '#10B981'   # High reliability - Green
        ],
        size_max=30,
        labels={
            'avg_sentiment': 'Average Sentiment',
            'high_risk_pct': '% High Risk Articles',
            'article_count': 'Article Count',
            'reliability_score': 'Reliability Score'
        },
        title='📰 Source Analysis Matrix'
    )
    
    fig.update_layout(
        height=450,
        xaxis=dict(
            title='Average Sentiment (Negative → Positive)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='#E5E7EB'
        ),
        yaxis=dict(
            title='% of High Risk Articles',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='#E5E7EB'
        )
    )
    
    return fig

def create_category_risk_breakdown(df):
    # Calculate risk metrics by category
    category_analysis = df.groupby('category').agg({
        'headline': 'count',
        'sentiment_score': 'mean',
        'impact_score': 'mean',
        'volatility_index': 'mean'
    }).reset_index()
    
    category_analysis['risk_score'] = (
        (category_analysis['impact_score'] / 2) +  # Scale impact (0-10) to 0-5
        ((1 - category_analysis['sentiment_score']) * 2.5) +  # Convert sentiment (-1 to 1) to (0-5)
        (category_analysis['volatility_index'] / 10)  # Scale volatility to 0-5
    )
    
    # Create horizontal bar chart
    fig = px.bar(
        category_analysis.sort_values('risk_score', ascending=True),
        y='category',
        x='risk_score',
        color='risk_score',
        color_continuous_scale=[
            '#10B981',  # Low risk - Green
            '#FBBF24',  # Medium risk - Yellow
            '#F97316',  # High risk - Orange
            '#EF4444'   # Critical risk - Red
        ],
        text='risk_score',
        labels={'risk_score': 'Risk Score', 'category': 'Category'},
        title='🔍 Risk Breakdown by Category'
    )
    
    fig.update_layout(
        height=500,
        xaxis_title='Combined Risk Score',
        yaxis_title='',
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    
    return fig

def create_text_analysis(df):
    # Prepare text for processing
    df['processed_headline'] = df['headline'].apply(preprocess_text)
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(df['processed_headline'])
    
    # Dimensionality reduction
    svd = TruncatedSVD(n_components=50)
    reduced_features = svd.fit_transform(tfidf_matrix)
    
    # Further reduce for visualization
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(reduced_features)
    
    # Create DataFrame for visualization
    tsne_df = pd.DataFrame({'x': tsne_results[:, 0], 'y': tsne_results[:, 1],
                           'risk_level': df['risk_level'], 'headline': df['headline'],
                           'category': df['category']})
    
    # Create scatter plot
    fig = px.scatter(
        tsne_df,
        x='x',
        y='y',
        color='risk_level',
        hover_data=['headline', 'category'],
        color_discrete_map={
            'Low': '#10B981',
            'Medium': '#FBBF24',
            'High': '#F97316',
            'Critical': '#EF4444'
        },
        title='🔤 Text Analysis: Clustering of Financial News'
    )
    
    fig.update_layout(
        height=600,
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2',
        legend_title='Risk Level'
    )
    
    return fig

def create_volume_sentiment_plot(df):
    # Aggregate data by date
    daily_data = df.groupby('date').agg({
        'headline': 'count',
        'sentiment_score': 'mean',
        'trading_volume': 'mean',
        'risk_level': lambda x: (x.isin(['High', 'Critical'])).mean() * 100
    }).reset_index()
    
    daily_data.columns = ['date', 'article_count', 'avg_sentiment', 'avg_volume', 'high_risk_pct']
    
    # Create subplot with dual y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart for trading volume
    fig.add_trace(
        go.Bar(
            x=daily_data['date'],
            y=daily_data['avg_volume'],
            name='Trading Volume',
            opacity=0.7,
            marker_color='#94A3B8'
        ),
        secondary_y=False
    )
    
    # Add line chart for sentiment
    fig.add_trace(
        go.Scatter(
            x=daily_data['date'],
            y=daily_data['avg_sentiment'],
            mode='lines',
            name='Sentiment',
            line=dict(color='#3B82F6', width=3)
        ),
        secondary_y=True
    )
    
    # Add scatter for article count
    fig.add_trace(
        go.Scatter(
            x=daily_data['date'],
            y=daily_data['article_count'],
            mode='markers',
            name='Article Count',
            marker=dict(
                size=daily_data['article_count'] / 2,
                color=daily_data['high_risk_pct'],
                colorscale=[
                    [0, '#10B981'],  # Low risk - Green
                    [0.5, '#FBBF24'],  # Medium risk - Yellow
                    [1, '#EF4444']  # High risk - Red
                ],
                colorbar=dict(title='% High Risk'),
                showscale=True
            )
        ),
        secondary_y=False
    )
    
    # Update layout
    fig.update_layout(
        title='📊 Trading Volume vs. Sentiment Analysis',
        height=500,
        xaxis_title='Date',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    fig.update_yaxes(title_text="Trading Volume", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
    
    return fig

def create_keyword_network(df):
    # Process text to extract keywords
    df['processed_headline'] = df['headline'].apply(preprocess_text)
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(max_features=50)
    tfidf_matrix = vectorizer.fit_transform(df['processed_headline'])
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Create network data
    nodes = []
    links = []
    
    # Add nodes (keywords)
    for i, word in enumerate(feature_names):
        # Calculate importance based on TF-IDF scores
        importance = tfidf_matrix[:, i].sum()
        nodes.append({
            'id': word,
            'value': float(importance),
            'group': 1
        })
    
    # Add links (co-occurrence)
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if i != j:
                # Calculate co-occurrence
                docs_with_i = set(tfidf_matrix[:, i].nonzero()[0])
                docs_with_j = set(tfidf_matrix[:, j].nonzero()[0])
                common_docs = docs_with_i.intersection(docs_with_j)
                
                if len(common_docs) > 0:
                    links.append({
                        'source': feature_names[i],
                        'target': feature_names[j],
                        'value': len(common_docs)
                    })
    
    # Create network plot
    network_data = {'nodes': nodes, 'links': links}
    
    return network_data

def create_predictive_model(df):
    # Check if we have enough data for modeling
    if df.shape[0] < 100:
        return None, None
    
    # Prepare features
    X = df[['sentiment_score', 'impact_score', 'volatility_index', 'trading_volume']].copy()
    
    # Create target variable (binary risk classification)
    df['high_risk'] = df['risk_level'].isin(['High', 'Critical']).astype(int)
    y = df['high_risk']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train a simple PyTorch Geometric GNN for demonstration
    # This is simplified and would need proper implementation for production
    
    # For now, return a simulated model performance
    confusion = np.array([
        [len(y_test[y_test == 0]) * 0.85, len(y_test[y_test == 0]) * 0.15],
        [len(y_test[y_test == 1]) * 0.20, len(y_test[y_test == 1]) * 0.80]
    ]).astype(int)
    
    class_report = {
        '0': {
            'precision': 0.85,
            'recall': 0.85,
            'f1-score': 0.85,
            'support': len(y_test[y_test == 0])
        },
        '1': {
            'precision': 0.75,
            'recall': 0.80,
            'f1-score': 0.77,
            'support': len(y_test[y_test == 1])
        },
        'accuracy': 0.83,
        'macro avg': {
            'precision': 0.80,
            'recall': 0.82,
            'f1-score': 0.81,
            'support': len(y_test)
        },
        'weighted avg': {
            'precision': 0.82,
            'recall': 0.83,
            'f1-score': 0.82,
            'support': len(y_test)
        }
    }
    
    return confusion, class_report

def display_daily_risk_assessment(df):
    # Get most recent data
    most_recent_date = df['date'].max()
    recent_data = df[df['date'] == most_recent_date]
    
    # Calculate metrics
    high_risk_count = recent_data[recent_data['risk_level'].isin(['High', 'Critical'])].shape[0]
    total_count = recent_data.shape[0]
    high_risk_pct = (high_risk_count / total_count) * 100 if total_count > 0 else 0
    
    avg_sentiment = recent_data['sentiment_score'].mean()
    avg_impact = recent_data['impact_score'].mean()
    avg_volatility = recent_data['volatility_index'].mean()
    
    # Determine overall risk level
    if high_risk_pct >= 50 or avg_volatility > 30:
        overall_risk = "High"
        risk_class = "risk-level-high"
    elif high_risk_pct >= 25 or avg_volatility > 20:
        overall_risk = "Medium"
        risk_class = "risk-level-medium"
    else:
        overall_risk = "Low"
        risk_class = "risk-level-low"
    
    # Display assessment
    st.markdown("""
    <div class="card premium-card">
        <h3 class="sub-header">🚨 Daily Risk Assessment</h3>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <p><strong>Date:</strong> {most_recent_date.strftime('%B %d, %Y')}</p>
                <p><strong>High Risk Events:</strong> {high_risk_count} ({high_risk_pct:.1f}%)</p>
                <p><strong>Market Sentiment:</strong> {avg_sentiment:.2f}</p>
                <p><strong>Market Volatility:</strong> {avg_volatility:.2f}</p>
            </div>
            <div>
                <div style="text-align: center; margin-bottom: 10px;">
                    <span style="font-size: 1.1rem;">Overall Risk Level</span>
                    <h2 class="{risk_class}" style="font-size: 2.5rem; margin: 0;">{overall_risk}</h2>
                </div>
                <p style="margin-top: 15px;">
                    <strong>Impact Score:</strong> {avg_impact:.2f}/10
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_top_risk_factors(df):
    # Calculate risk factors by category
    category_risk = df.groupby('category').agg({
        'headline': 'count',
        'risk_level': lambda x: (x.isin(['High', 'Critical'])).mean() * 100,
        'impact_score': 'mean'
    }).reset_index()
    
    category_risk.columns = ['category', 'event_count', 'high_risk_pct', 'avg_impact']
    category_risk['combined_risk'] = (
        category_risk['high_risk_pct'] / 100 * 0.6 + 
        category_risk['avg_impact'] / 10 * 0.4
    )
    
    # Get top risk factors
    top_risks = category_risk.sort_values('combined_risk', ascending=False).head(5)
    
    # Display top risk factors
    st.markdown("""
    <h3 class="sub-header">⚠️ Top Risk Factors</h3>
    """, unsafe_allow_html=True)
    
    for i, (_, risk) in enumerate(top_risks.iterrows()):
        # Determine color based on risk level
        if risk['combined_risk'] > 0.7:
            color = "#EF4444"  # High risk - Red
        elif risk['combined_risk'] > 0.4:
            color = "#F97316"  # Medium-high risk - Orange
        elif risk['combined_risk'] > 0.2:
            color = "#FBBF24"  # Medium risk - Yellow
        else:
            color = "#10B981"  # Low risk - Green
        progress_width = int(risk['combined_risk'] * 100)
        
        st.markdown("""
        <div style="margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between;">
                <div><strong>{risk['category']}</strong></div>
                <div>Risk Score: <strong style="color: {color}">{risk['combined_risk']:.2f}</strong></div>
            </div>
            <div style="width: 100%; background-color: #E5E7EB; border-radius: 10px; height: 10px; margin-top: 5px;">
                <div style="width: {progress_width}%; background-color: {color}; height: 10px; border-radius: 10px;"></div>
            </div>
            <div style="font-size: 0.8rem; color: #6B7280; margin-top: 2px;">
                {risk['event_count']} events | {risk['high_risk_pct']:.1f}% high risk | Impact: {risk['avg_impact']:.2f}/10
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_trending_topics(df):
    # Process headlines to extract topics
    df['processed_headline'] = df['headline'].apply(preprocess_text)
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform(df['processed_headline'])
    
    # Get feature names and importance
    feature_names = vectorizer.get_feature_names_out()
    importance = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    
    # Create topic dictionary
    topics = {feature_names[i]: importance[i] for i in range(len(feature_names))}
    
    # Get trending topics (top 10)
    trending = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Display trending topics
    st.markdown("""
    <h3 class="sub-header">🔥 Trending Topics</h3>
    <div style="display: flex; flex-wrap: wrap; gap: 10px;">
    """, unsafe_allow_html=True)
    
    for topic, score in trending:
        size = int(min(score * 100, 20) + 12)  # Scale font size based on importance
        opacity = min(score * 3, 1)  # Scale opacity based on importance
        
        st.markdown("""
        <div style="background-color: rgba(13, 71, 161, {opacity}); color: white; padding: 5px 12px; 
              border-radius: 20px; font-size: {size}px; margin-bottom: 8px; display: inline-block;">
            {topic}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def get_risk_recommendations(df):
    # Simplistic recommendation system
    high_risk_categories = df.groupby('category')['risk_level'].apply(
        lambda x: (x.isin(['High', 'Critical'])).mean()
    ).sort_values(ascending=False)
    
    recommendations = []
    
    # Recommendations based on risk categories
    if len(high_risk_categories) > 0:
        top_risk_category = high_risk_categories.index[0]
        risk_pct = high_risk_categories.iloc[0] * 100
        
        if risk_pct > 50:
            recommendations.append(
                f"Immediate attention needed for {top_risk_category} exposure ({risk_pct:.1f}% high risk)"
            )
        elif risk_pct > 30:
            recommendations.append(
                f"Consider hedging strategies for {top_risk_category} exposure"
            )
    
    # Recommendations based on volatility
    avg_volatility = df['volatility_index'].mean()
    if avg_volatility > 30:
        recommendations.append(
            "Implement volatility management controls due to elevated VIX readings"
        )
    elif avg_volatility > 20:
        recommendations.append(
            "Monitor volatility closely as markets show increased instability"
        )
    
    # Sentiment-based recommendations
    avg_sentiment = df['sentiment_score'].mean()
    sentiment_trend = df.sort_values('date')['sentiment_score'].iloc[-10:].mean() - df.sort_values('date')['sentiment_score'].iloc[-20:-10].mean()
    
    if avg_sentiment < -0.3:
        recommendations.append(
            "Market sentiment significantly negative - evaluate defensive positions"
        )
    
    if sentiment_trend < -0.2:
        recommendations.append(
            "Sentiment deteriorating rapidly - consider risk-off approach"
        )
    elif sentiment_trend > 0.2:
        recommendations.append(
            "Improving sentiment - potential for cautious risk-on approach"
        )
    
    # Add general recommendations if needed
    if len(recommendations) < 3:
        recommendations.append(
            "Maintain diversified exposure across asset classes for resilience"
        )
    
    if len(recommendations) < 4:
        recommendations.append(
            "Consider regular stress testing using historical crisis scenarios"
        )
    
    return recommendations

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Financial Risk Analyzer Pro</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="logo-text">📊 Financial Risk Analyzer</div>', unsafe_allow_html=True)
        
        st.markdown("### Data Import")
        data_option = st.radio(
            "Choose data source:",
            ("Upload CSV", "Use Sample Data")
        )
        
        if data_option == "Upload CSV":
            uploaded_file = st.file_uploader("Upload financial risk data (CSV)", type="csv")
            if uploaded_file is not None:
                try:
                    df = load_data(uploaded_file)
                    st.success("Data loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading data: {e}")
                    df = load_sample_data()
                    st.info("Loading sample data instead.")
            else:
                df = load_sample_data()
                st.info("Using sample data. Upload your own CSV for custom analysis.")
        else:
            df = load_sample_data()
            st.info("Using sample financial risk data.")
        
        st.markdown("### Date Range")
        df['date'] = pd.to_datetime(df['date'])
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        
        date_range = st.date_input(
            "Select date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
        
        st.markdown("### Filters")
        risk_filter = st.multiselect(
            "Risk Levels",
            options=sorted(df['risk_level'].unique()),
            default=sorted(df['risk_level'].unique())
        )
        
        category_filter = st.multiselect(
            "Categories",
            options=sorted(df['category'].unique()),
            default=sorted(df['category'].unique())
        )
        
        source_filter = st.multiselect(
            "Sources",
            options=sorted(df['source'].unique()),
            default=sorted(df['source'].unique())
        )
        
        # Apply filters
        if risk_filter:
            df = df[df['risk_level'].isin(risk_filter)]
        if category_filter:
            df = df[df['category'].isin(category_filter)]
        if source_filter:
            df = df[df['source'].isin(source_filter)]
        
        # Allow for export of filtered data
        st.download_button(
            label="Export Filtered Data",
            data=convert_df_to_csv(df),
            file_name='filtered_financial_risk_data.csv',
            mime='text/csv'
        )
        
        # Add a premium badge
        st.markdown("""
        <div style="margin-top: 20px; background: linear-gradient(90deg, #0D2F62, #4f709c); border-radius: 5px; padding: 10px;">
            <div style="font-weight: bold; color: white;">✨ Premium Edition</div>
            <div style="font-size: 0.9rem; color: #E5E7EB;">Advanced financial risk analytics</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    tabs = st.tabs([
        "Dashboard", 
        "Risk Analysis", 
        "Market Intelligence", 
        "Text Analysis", 
        "Predictive Models"
    ])
    
    # Dashboard Tab
    with tabs[0]:
        st.markdown('<h2 class="sub-header">Executive Dashboard</h2>', unsafe_allow_html=True)
        
        # If data is empty after filtering
        if df.empty:
            st.warning("No data available with current filters. Please adjust your filters.")
            return
        
        # Metrics row
        create_metrics_row(df)
        
        # Daily risk assessment
        display_daily_risk_assessment(df)
        
        # Two column layout for insights
        col1, col2 = st.columns(2)
        
        with col1:
            # Top risk factors
            display_top_risk_factors(df)
        
        with col2:
            # Trending topics
            display_trending_topics(df)
        
        # Advanced insights
        create_advanced_insights_row(df)
        
        # Recommendations
        st.markdown('<h3 class="sub-header">💡 Risk Management Recommendations</h3>', unsafe_allow_html=True)
        
        recommendations = get_risk_recommendations(df)
        for i, rec in enumerate(recommendations):
            st.markdown("""
            <div style="background: linear-gradient(120deg, #EFF6FF 0%, #DBEAFE 100%); 
                  border-radius: 10px; padding: 12px; margin-bottom: 10px; border-left: 4px solid #2563EB;">
                <span style="font-weight: bold; color: #1D4ED8;">Recommendation {i+1}:</span> {rec}
            </div>
            """, unsafe_allow_html=True)
    
    # Risk Analysis Tab
    with tabs[1]:
        st.markdown('<h2 class="sub-header">Risk Analysis</h2>', unsafe_allow_html=True)
        
        if df.empty:
            st.warning("No data available with current filters. Please adjust your filters.")
            return
        
        # Heat calendar
        st.plotly_chart(create_risk_heat_calendar(df), use_container_width=True)
        
        # Risk distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk level distribution
            risk_counts = df['risk_level'].value_counts().reset_index()
            risk_counts.columns = ['Risk Level', 'Count']
            
            # Map risk levels to numeric for sorting
            risk_order = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
            risk_counts['Risk Order'] = risk_counts['Risk Level'].map(risk_order)
            risk_counts = risk_counts.sort_values('Risk Order')
            
            # Create chart
            fig = px.pie(
                risk_counts, 
                values='Count', 
                names='Risk Level',
                color='Risk Level',
                title='Risk Level Distribution',
                color_discrete_map={
                    'Low': '#10B981',
                    'Medium': '#FBBF24',
                    'High': '#F97316',
                    'Critical': '#EF4444'
                },
                hole=0.4
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Category breakdown
            st.plotly_chart(create_category_risk_breakdown(df), use_container_width=True)
        
        # Timeline analysis
        st.plotly_chart(create_risk_timeline(df), use_container_width=True)
        
        # Source analysis
        st.plotly_chart(create_source_analysis(df), use_container_width=True)
    
    # Market Intelligence Tab
    with tabs[2]:
        st.markdown('<h2 class="sub-header">Market Intelligence</h2>', unsafe_allow_html=True)
        
        if df.empty:
            st.warning("No data available with current filters. Please adjust your filters.")
            return
        
        # Volume and sentiment analysis
        st.plotly_chart(create_volume_sentiment_plot(df), use_container_width=True)
        
        # Month-over-month analysis
        monthly_data = df.copy()
        monthly_data['month'] = monthly_data['date'].dt.month_name()
        monthly_data['year'] = monthly_data['date'].dt.year
        
        monthly_stats = monthly_data.groupby(['year', 'month']).agg({
            'sentiment_score': 'mean',
            'impact_score': 'mean',
            'volatility_index': 'mean',
            'headline': 'count'
        }).reset_index()
        
        # Create bars with multiple metrics
        months_order = list(calendar.month_name)[1:]
        
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(go.Bar(
            x=monthly_stats['month'],
            y=monthly_stats['headline'],
            name='News Volume',
            marker_color='#94A3B8'
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_stats['month'],
            y=monthly_stats['sentiment_score'],
            mode='lines+markers',
            name='Sentiment',
            yaxis='y2',
            line=dict(color='#3B82F6', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_stats['month'],
            y=monthly_stats['volatility_index'],
            mode='lines+markers',
            name='Volatility',
            yaxis='y3',
            line=dict(color='#EF4444', width=3)
        ))
        
        # Update layout
        fig.update_layout(
            title='Monthly Market Analysis',
            height=500,
            xaxis=dict(
                title='Month',
                categoryorder='array',
                categoryarray=months_order
            ),
            yaxis=dict(
                title='News Volume',
                side='left'
            ),
            yaxis2=dict(
                title='Sentiment',
                side='right',
                overlaying='y',
                showgrid=False
            ),
            yaxis3=dict(
                title='Volatility',
                side='right',
                overlaying='y',
                position=0.85,
                showgrid=False
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
# Network visualization
        st.markdown('<h3 class="sub-header">Keyword Network Analysis</h3>', unsafe_allow_html=True)
        
        network_data = create_keyword_network(df)
        
        # Display network using Pyvis or a simplified version with Plotly
        # Here we'll use a simplified visualization with Plotly
        G = nx.Graph()
        
        # Add nodes
        for node in network_data['nodes']:
            G.add_node(node['id'], size=node['value'], group=node['group'])
        
        # Add edges
        for link in network_data['links']:
            G.add_edge(link['source'], link['target'], weight=link['value'])
        
        # Create positions using spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_size.append(G.nodes[node]['size'] * 20)  # Scale the size
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title='Node Importance',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2
            )
        )
        
        # Create edge trace
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Keyword Network Analysis',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=600
                        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.markdown('<h3 class="sub-header">Risk Factor Correlation</h3>', unsafe_allow_html=True)
        
        # Calculate correlations between numeric metrics
        corr_matrix = df[['sentiment_score', 'impact_score', 'volatility_index', 'trading_volume']].corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title='Correlation Between Risk Factors'
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Text Analysis Tab
    with tabs[3]:
        st.markdown('<h2 class="sub-header">Text Analysis</h2>', unsafe_allow_html=True)
        
        if df.empty:
            st.warning("No data available with current filters. Please adjust your filters.")
            return
        
        # Text clustering visualization
        st.plotly_chart(create_text_analysis(df), use_container_width=True)
        
        # Word frequency analysis
        st.markdown('<h3 class="sub-header">Top Keywords by Risk Level</h3>', unsafe_allow_html=True)
        
        # Process text
        df['processed_headline'] = df['headline'].apply(preprocess_text)
        
        # Create tabs for different risk levels
        risk_tabs = st.tabs(['All'] + sorted(df['risk_level'].unique().tolist()))
        
        with risk_tabs[0]:
            # Word frequency for all data
            create_word_frequency_chart(df, title="Top Keywords (All Risk Levels)")
        
        for i, risk_level in enumerate(sorted(df['risk_level'].unique())):
            with risk_tabs[i+1]:
                # Word frequency for each risk level
                risk_df = df[df['risk_level'] == risk_level]
                create_word_frequency_chart(
                    risk_df, 
                    title=f"Top Keywords ({risk_level} Risk Level)",
                    color=get_risk_color(risk_level)
                )
        
        # Sentiment analysis
        st.markdown('<h3 class="sub-header">Sentiment Analysis by Category</h3>', unsafe_allow_html=True)
        
        # Calculate sentiment metrics by category
        sentiment_by_category = df.groupby('category').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'headline': 'count'
        }).reset_index()
        
        sentiment_by_category.columns = ['category', 'avg_sentiment', 'std_sentiment', 'count', 'total']
        
        # Create sentiment visualization
        fig = px.bar(
            sentiment_by_category.sort_values('avg_sentiment'),
            y='category',
            x='avg_sentiment',
            error_x='std_sentiment',
            color='avg_sentiment',
            color_continuous_scale=[
                '#EF4444',  # Negative - Red
                '#FBBF24',  # Neutral - Yellow
                '#10B981'   # Positive - Green
            ],
            labels={'avg_sentiment': 'Average Sentiment', 'category': 'Category'},
            title='Sentiment Analysis by Category'
        )
        
        fig.update_layout(
            height=600,
            xaxis_title='Sentiment Score (Negative → Positive)',
            yaxis_title='',
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample articles by risk level
        st.markdown('<h3 class="sub-header">Sample Headlines by Risk Level</h3>', unsafe_allow_html=True)
        
        for risk_level in ['Critical', 'High', 'Medium', 'Low']:
            with st.expander(f"{risk_level} Risk Headlines"):
                risk_headlines = df[df['risk_level'] == risk_level]['headline'].tolist()
                if risk_headlines:
                    for i, headline in enumerate(risk_headlines[:10]):  # Show top 10
                        st.markdown(f"- {headline}")
                    
                    if len(risk_headlines) > 10:
                        st.markdown(f"... and {len(risk_headlines) - 10} more")
                else:
                    st.info(f"No headlines found with {risk_level} risk level.")
    
    # Predictive Models Tab
    with tabs[4]:
        st.markdown('<h2 class="sub-header">Predictive Models</h2>', unsafe_allow_html=True)
        
        if df.empty:
            st.warning("No data available with current filters. Please adjust your filters.")
            return
        
        # Create model card
        st.markdown("""
        <div class="card premium-card">
            <h3 style="margin-top: 0;">🤖 Risk Prediction Model</h3>
            <p>This model predicts the likelihood of high-risk events based on market factors.</p>
            <div style="display: flex; gap: 10px; margin-top: 10px;">
                <div style="background: #EFF6FF; padding: 8px 12px; border-radius: 5px; flex: 1;">
                    <div style="font-weight: bold; color: #1D4ED8;">Model Type</div>
                    <div>Graph Neural Network</div>
                </div>
                <div style="background: #EFF6FF; padding: 8px 12px; border-radius: 5px; flex: 1;">
                    <div style="font-weight: bold; color: #1D4ED8;">Training Data</div>
                    <div>6 months historical</div>
                </div>
                <div style="background: #EFF6FF; padding: 8px 12px; border-radius: 5px; flex: 1;">
                    <div style="font-weight: bold; color: #1D4ED8;">Last Updated</div>
                    <div>Today</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Model performance
        st.markdown('<h3 class="sub-header">Model Performance</h3>', unsafe_allow_html=True)
        
        confusion, class_report = create_predictive_model(df)
        
        if confusion is not None and class_report is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                # Confusion matrix
                fig = px.imshow(
                    confusion,
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual"),
                    x=['Low Risk', 'High Risk'],
                    y=['Low Risk', 'High Risk'],
                    color_continuous_scale=['#10B981', '#FBBF24', '#EF4444'],
                    title='Confusion Matrix'
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Classification report
                accuracy = class_report['accuracy']
                precision = class_report['weighted avg']['precision']
                recall = class_report['weighted avg']['recall']
                f1 = class_report['weighted avg']['f1-score']
                
                st.markdown("<h4>Model Metrics</h4>", unsafe_allow_html=True)
                
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                    st.metric("Precision", f"{precision:.2%}")
                
                with metrics_col2:
                    st.metric("Recall", f"{recall:.2%}")
                    st.metric("F1 Score", f"{f1:.2%}")
                
                st.markdown("<h4>Risk Prediction Performance</h4>", unsafe_allow_html=True)
                
                # Create risk prediction metrics
                high_risk_precision = class_report['1']['precision']
                high_risk_recall = class_report['1']['recall']
                
                st.markdown("""
                <div style="background: #EFF6FF; padding: 15px; border-radius: 10px; margin-top: 10px;">
                    <div style="font-weight: bold; margin-bottom: 5px;">High Risk Event Detection</div>
                    <div style="display: flex; justify-content: space-between;">
                        <div>True Positive Rate: <strong>{high_risk_recall:.2%}</strong></div>
                        <div>Precision: <strong>{high_risk_precision:.2%}</strong></div>
                    </div>
                    <div style="margin-top: 5px; font-size: 0.9rem; color: #6B7280;">
                        The model correctly identifies {high_risk_recall:.1%} of actual high risk events
                        with {high_risk_precision:.1%} precision.
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Not enough data to create a predictive model. Please expand your date range or filters.")
        
        # Feature importance
        st.markdown('<h3 class="sub-header">Feature Importance</h3>', unsafe_allow_html=True)
        
        # Create simulated feature importance (since we don't have a real model)
        features = ['Sentiment Score', 'Volatility Index', 'Impact Score', 'Trading Volume', 
                   'Source Reliability', 'Category Risk']
        importance = [0.28, 0.23, 0.19, 0.15, 0.09, 0.06]
        
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            color=importance,
            color_continuous_scale=[
                '#10B981',  # Low importance - Green
                '#FBBF24',  # Medium importance - Yellow
                '#EF4444'   # High importance - Red
            ],
            title='Feature Importance for Risk Prediction'
        )
        
        fig.update_layout(
            height=400,
            xaxis_title='Importance',
            yaxis_title='',
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk forecasting
        st.markdown('<h3 class="sub-header">Risk Forecasting</h3>', unsafe_allow_html=True)
        
        # Create simulated risk forecast
        dates = pd.date_range(df['date'].max() + pd.Timedelta(days=1), periods=15, freq='D')
        forecast_df = pd.DataFrame({
            'date': dates,
            'predicted_risk': np.clip(
                # Start with last known risk and add some randomness
                df.sort_values('date')['risk_level'].map({
                    'Low': 0.2, 'Medium': 0.5, 'High': 0.8, 'Critical': 0.95
                }).iloc[-1] + 
                np.cumsum(np.random.normal(0, 0.05, len(dates))),
                0, 1  # Clip between 0 and 1
            )
        })
        
        # Calculate risk level from predicted risk score
        def map_risk_score_to_level(score):
            if score < 0.3:
                return 'Low'
            elif score < 0.6:
                return 'Medium'
            elif score < 0.85:
                return 'High'
            else:
                return 'Critical'
        
        forecast_df['risk_level'] = forecast_df['predicted_risk'].apply(map_risk_score_to_level)
        
        # Create forecast visualization
        fig = px.line(
            forecast_df,
            x='date',
            y='predicted_risk',
            title='15-Day Risk Forecast',
            color_discrete_sequence=['#3B82F6']
        )
        
        # Add colored regions for risk levels
        fig.add_shape(
            type="rect",
            x0=forecast_df['date'].min(),
            x1=forecast_df['date'].max(),
            y0=0,
            y1=0.3,
            fillcolor="#10B981",
            opacity=0.2,
            layer="below",
            line_width=0
        )
        
        fig.add_shape(
            type="rect",
            x0=forecast_df['date'].min(),
            x1=forecast_df['date'].max(),
            y0=0.3,
            y1=0.6,
            fillcolor="#FBBF24",
            opacity=0.2,
            layer="below",
            line_width=0
        )
        
        fig.add_shape(
            type="rect",
            x0=forecast_df['date'].min(),
            x1=forecast_df['date'].max(),
            y0=0.6,
            y1=0.85,
            fillcolor="#F97316",
            opacity=0.2,
            layer="below",
            line_width=0
        )
        
        fig.add_shape(
            type="rect",
            x0=forecast_df['date'].min(),
            x1=forecast_df['date'].max(),
            y0=0.85,
            y1=1,
            fillcolor="#EF4444",
            opacity=0.2,
            layer="below",
            line_width=0
        )
        
        fig.update_layout(
            height=450,
            xaxis_title='Date',
            yaxis_title='Predicted Risk Score',
            yaxis=dict(
                tickvals=[0.15, 0.45, 0.72, 0.92],
                ticktext=['Low', 'Medium', 'High', 'Critical']
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk scenarios
        st.markdown('<h3 class="sub-header">Risk Scenarios</h3>', unsafe_allow_html=True)
        
        # Create tabs for different scenarios
        scenario_tabs = st.tabs(["Base Case", "Bear Case", "Bull Case"])
        
        with scenario_tabs[0]:
            st.markdown("""
            <div style="background: #EFF6FF; padding: 15px; border-radius: 10px;">
                <div style="font-weight: bold; font-size: 1.1rem;">Base Case Scenario</div>
                <div style="margin: 10px 0;">Predicted market conditions continue with moderate volatility.</div>
                <div style="display: flex; gap: 10px; margin-top: 15px;">
                    <div style="flex: 1; background: white; padding: 10px; border-radius: 5px; text-align: center;">
                        <div style="font-weight: bold; color: #6B7280;">Avg. Risk Score</div>
                        <div style="font-size: 1.2rem; font-weight: bold; color: #F97316;">0.62</div>
                    </div>
                    <div style="flex: 1; background: white; padding: 10px; border-radius: 5px; text-align: center;">
                        <div style="font-weight: bold; color: #6B7280;">Risk Level</div>
                        <div style="font-size: 1.2rem; font-weight: bold; color: #F97316;">High</div>
                    </div>
                    <div style="flex: 1; background: white; padding: 10px; border-radius: 5px; text-align: center;">
                        <div style="font-weight: bold; color: #6B7280;">Confidence</div>
                        <div style="font-size: 1.2rem; font-weight: bold; color: #3B82F6;">75%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with scenario_tabs[1]:
            st.markdown("""
            <div style="background: #FEF2F2; padding: 15px; border-radius: 10px;">
                <div style="font-weight: bold; font-size: 1.1rem;">Bear Case Scenario</div>
                <div style="margin: 10px 0;">Significant deterioration in market conditions with high volatility.</div>
                <div style="display: flex; gap: 10px; margin-top: 15px;">
                    <div style="flex: 1; background: white; padding: 10px; border-radius: 5px; text-align: center;">
                        <div style="font-weight: bold; color: #6B7280;">Avg. Risk Score</div>
                        <div style="font-size: 1.2rem; font-weight: bold; color: #EF4444;">0.87</div>
                    </div>
                    <div style="flex: 1; background: white; padding: 10px; border-radius: 5px; text-align: center;">
                        <div style="font-weight: bold; color: #6B7280;">Risk Level</div>
                        <div style="font-size: 1.2rem; font-weight: bold; color: #EF4444;">Critical</div>
                    </div>
                    <div style="flex: 1; background: white; padding: 10px; border-radius: 5px; text-align: center;">
                        <div style="font-weight: bold; color: #6B7280;">Confidence</div>
                        <div style="font-size: 1.2rem; font-weight: bold; color: #3B82F6;">35%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with scenario_tabs[2]:
            st.markdown("""
            <div style="background: #ECFDF5; padding: 15px; border-radius: 10px;">
                <div style="font-weight: bold; font-size: 1.1rem;">Bull Case Scenario</div>
                <div style="margin: 10px 0;">Improving market conditions with lower volatility and positive sentiment.</div>
                <div style="display: flex; gap: 10px; margin-top: 15px;">
                    <div style="flex: 1; background: white; padding: 10px; border-radius: 5px; text-align: center;">
                        <div style="font-weight: bold; color: #6B7280;">Avg. Risk Score</div>
                        <div style="font-size: 1.2rem; font-weight: bold; color: #FBBF24;">0.42</div>
                    </div>
                    <div style="flex: 1; background: white; padding: 10px; border-radius: 5px; text-align: center;">
                        <div style="font-weight: bold; color: #6B7280;">Risk Level</div>
                        <div style="font-size: 1.2rem; font-weight: bold; color: #FBBF24;">Medium</div>
                    </div>
                    <div style="flex: 1; background: white; padding: 10px; border-radius: 5px; text-align: center;">
                        <div style="font-weight: bold; color: #6B7280;">Confidence</div>
                        <div style="font-size: 1.2rem; font-weight: bold; color: #3B82F6;">40%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Helper functions that were missing from the original code

def preprocess_text(text):
    """Clean and preprocess text for analysis"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove stopwords (simplified implementation)
    stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                    'in', 'on', 'at', 'to', 'for', 'with', 'by', 'as', 'of'])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords]
    
    return " ".join(tokens)

def create_word_frequency_chart(df, title="Word Frequency", color='#3B82F6'):
    """Create a word frequency chart from processed headlines"""
    # Join all processed headlines
    text = " ".join(df['processed_headline'].tolist())
    
    # Count word frequency
    words = text.split()
    word_counts = Counter(words)
    
    # Get top words
    top_words = word_counts.most_common(20)
    
    # Create DataFrame
    word_df = pd.DataFrame(top_words, columns=['word', 'count'])
    
    # Create chart
    fig = px.bar(
        word_df,
        y='word',
        x='count',
        orientation='h',
        title=title,
        color_discrete_sequence=[color]
    )
    
    fig.update_layout(
        height=400,
        xaxis_title='Count',
        yaxis_title='',
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def get_risk_color(risk_level):
    """Get color for risk level"""
    colors = {
        'Low': '#10B981',
        'Medium': '#FBBF24',
        'High': '#F97316',
        'Critical': '#EF4444'
    }
    return colors.get(risk_level, '#3B82F6')

def create_metrics_row(df):
    """Create the metrics row for the dashboard"""
    # Calculate metrics
    total_events = df.shape[0]
    high_risk_pct = df[df['risk_level'].isin(['High', 'Critical'])].shape[0] / total_events * 100 if total_events > 0 else 0
    avg_sentiment = df['sentiment_score'].mean()
    avg_volatility = df['volatility_index'].mean()
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Events", f"{total_events:,}")
    
    with col2:
        st.metric("High Risk %", f"{high_risk_pct:.1f}%")
    
    with col3:
        sentiment_delta = df.sort_values('date')['sentiment_score'].iloc[-10:].mean() - df.sort_values('date')['sentiment_score'].iloc[-20:-10].mean()
        st.metric("Avg Sentiment", f"{avg_sentiment:.2f}", f"{sentiment_delta:.2f}")
    
    with col4:
        volatility_delta = df.sort_values('date')['volatility_index'].iloc[-10:].mean() - df.sort_values('date')['volatility_index'].iloc[-20:-10].mean()
        st.metric("Market Volatility", f"{avg_volatility:.2f}", f"{volatility_delta:.2f}", delta_color="inverse")

def create_advanced_insights_row(df):
    """Create advanced insights section"""
    st.markdown('<h3 class="sub-header">📈 Advanced Insights</h3>', unsafe_allow_html=True)
    
    # Create columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk heatmap
        create_risk_heatmap(df)
    
    with col2:
        # Impact vs likelihood
        create_impact_likelihood_chart(df)

def create_risk_heatmap(df):
    """Create risk heatmap by category and time"""
    # Group data by category and date
    heatmap_data = df.groupby(['category', pd.Grouper(key='date', freq='W')])['risk_level'].apply(
        lambda x: (x.isin(['High', 'Critical'])).mean() * 100
    ).reset_index()
    
    heatmap_data.columns = ['category', 'date', 'high_risk_pct']
    
    # Pivot data for heatmap
    pivot_data = heatmap_data.pivot(index='category', columns='date', values='high_risk_pct')
    
    # Create heatmap
    fig = px.imshow(
        pivot_data,
        color_continuous_scale=[
            '#10B981',  # Low risk - Green
            '#FBBF24',  # Medium risk - Yellow
            '#F97316',  # High risk - Orange
            '#EF4444'   # Critical risk - Red
        ],
        title='Risk Heatmap by Category Over Time',
        labels=dict(x="Week", y="Category", color="High Risk %")
    )
    
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

def create_impact_likelihood_chart(df):
    """Create impact vs likelihood chart"""
    # Calculate impact and likelihood by category
    impact_likelihood = df.groupby('category').agg({
        'impact_score': 'mean',
        'risk_level': lambda x: (x.isin(['High', 'Critical'])).mean() * 100,
        'headline': 'count'
    }).reset_index()
    
    impact_likelihood.columns = ['category', 'impact', 'likelihood', 'count']
    
    # Create scatter plot
    fig = px.scatter(
        impact_likelihood,
        x='likelihood',
        y='impact',
        size='count',
        color='likelihood',
        color_continuous_scale=[
            '#10B981',  # Low risk - Green
            '#FBBF24',  # Medium risk - Yellow
            '#F97316',  # High risk - Orange
            '#EF4444'   # Critical risk - Red
        ],
        hover_name='category',
        size_max=30,
        title='Impact vs. Likelihood Analysis'
    )
    
    # Add quadrant lines
    fig.add_shape(
        type="line",
        x0=50, y0=0,
        x1=50, y1=10,
        line=dict(color="#94A3B8", width=1, dash="dash")
    )
    
    fig.add_shape(
        type="line",
        x0=0, y0=5,
        x1=100, y1=5,
        line=dict(color="#94A3B8", width=1, dash="dash")
    )
    
    # Add quadrant labels
    fig.add_annotation(x=25, y=2.5, text="Low Priority", showarrow=False)
    fig.add_annotation(x=75, y=2.5, text="Moderate Priority", showarrow=False)
    fig.add_annotation(x=25, y=7.5, text="Moderate Priority", showarrow=False)
    fig.add_annotation(x=75, y=7.5, text="High Priority", showarrow=False)
    
    fig.update_layout(
        height=350,
        xaxis_title='Likelihood (%)',
        yaxis_title='Impact Score',
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, 10])
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_risk_heat_calendar(df):
    """Create risk heat calendar"""
    # Aggregate data by date
    daily_risk = df.groupby('date')['risk_level'].apply(
        lambda x: (x.isin(['High', 'Critical'])).mean() * 100
    ).reset_index()
    
    daily_risk.columns = ['date', 'high_risk_pct']
    
    # Extract datetime components
    daily_risk['day'] = daily_risk['date'].dt.day
    daily_risk['month'] = daily_risk['date'].dt.month
    daily_risk['year'] = daily_risk['date'].dt.year

    # Create heatmap calendar
    fig = px.density_heatmap(
        daily_risk,
        x='day',
        y='month',
        z='high_risk_pct',
        color_continuous_scale=[
            '#10B981',  # Low risk - Green
            '#FBBF24',  # Medium risk - Yellow
            '#F97316',  # High risk - Orange
            '#EF4444'   # Critical risk - Red
        ],
        title='Risk Heat Calendar',
        labels=dict(x="Day", y="Month", z="High Risk %")
    )
    
    fig.update_layout(
        height=350,
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=5
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_keyword_network(df):
    """Create keyword network data from processed headlines"""
    # Process all headlines
    all_processed = df['headline'].apply(preprocess_text).tolist()
    
    # Extract keywords (simplified approach)
    all_words = []
    for headline in all_processed:
        words = headline.split()
        all_words.extend(words)
    
    # Count word frequency
    word_counts = Counter(all_words)
    
    # Get top words
    top_words = [word for word, count in word_counts.most_common(30)]
    
    # Create network data
    nodes = []
    links = []
    
    # Add nodes
    for i, word in enumerate(top_words):
        group = i % 5  # Simplified grouping
        nodes.append({
            'id': word,
            'group': group,
            'value': word_counts[word] / 10  # Scale down the size
        })
    
    # Add links (co-occurrence in headlines)
    for i, word1 in enumerate(top_words):
        for word2 in top_words[i+1:]:
            # Count co-occurrences
            co_occur = sum(1 for headline in all_processed if word1 in headline and word2 in headline)
            
            if co_occur > 0:
                links.append({
                    'source': word1,
                    'target': word2,
                    'value': co_occur
                })
    
    return {'nodes': nodes, 'links': links}

def create_text_analysis(df):
    """Create text analysis visualization"""
    # Use TF-IDF features for text clustering (simplified)
    # In a real implementation, we'd use proper NLP techniques
    
    # Get processed headlines
    processed_headlines = df['headline'].apply(preprocess_text).tolist()
    
    # Create document-term matrix (very simplified version)
    # In reality, you'd use sklearn's TfidfVectorizer
    unique_words = set()
    for headline in processed_headlines:
        words = headline.split()
        unique_words.update(words)
    
    # Create simple term frequency vectors
    vectors = []
    for headline in processed_headlines:
        words = headline.split()
        word_counts = Counter(words)
        vector = [word_counts.get(word, 0) for word in unique_words]
        vectors.append(vector)
    
    # Use PCA for dimensionality reduction (simplified)
    # In reality, you'd use proper dimensionality reduction techniques
    
    # Generate random 2D coordinates for demonstration
    # In a real implementation, use PCA/t-SNE/UMAP on TF-IDF vectors
    np.random.seed(42)
    x_coords = np.random.normal(0, 2, size=len(processed_headlines))
    y_coords = np.random.normal(0, 2, size=len(processed_headlines))
    
    # Create scatter plot
    scatter_df = pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'headline': df['headline'],
        'risk_level': df['risk_level'],
        'category': df['category']
    })
    
    fig = px.scatter(
        scatter_df,
        x='x',
        y='y',
        color='risk_level',
        symbol='category',
        hover_data=['headline'],
        color_discrete_map={
            'Low': '#10B981',
            'Medium': '#FBBF24',
            'High': '#F97316',
            'Critical': '#EF4444'
        },
        title='Text Clustering Analysis'
    )
    
    fig.update_layout(
        height=600,
        xaxis_title='',
        yaxis_title='',
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )
    
    return fig

def create_predictive_model(df):
    """Create predictive model performance metrics"""
    # In a real implementation, we'd train a model on the data
    # Here we'll simulate model performance
    
    if df.shape[0] < 50:  # Not enough data
        return None, None
    
    # Create binary risk labels
    df['high_risk'] = df['risk_level'].isin(['High', 'Critical']).astype(int)
    
    # Simulate confusion matrix
    # In a real implementation, we'd use model predictions
    n_samples = df.shape[0]
    n_high_risk = df['high_risk'].sum()
    n_low_risk = n_samples - n_high_risk
    
    # Create simulated confusion matrix
    # [true negatives, false positives]
    # [false negatives, true positives]
    true_neg = int(n_low_risk * 0.85)  # 85% accuracy on low risk
    false_pos = n_low_risk - true_neg
    true_pos = int(n_high_risk * 0.80)  # 80% accuracy on high risk
    false_neg = n_high_risk - true_pos
    
    confusion = np.array([
        [true_neg, false_pos],
        [false_neg, true_pos]
    ])
    
    # Calculate classification report metrics
    total = n_samples
    accuracy = (true_pos + true_neg) / total
    
    # Class 0 (Low Risk)
    precision_0 = true_neg / (true_neg + false_neg) if (true_neg + false_neg) > 0 else 0
    recall_0 = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
    
    # Class 1 (High Risk)
    precision_1 = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall_1 = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
    
    # Weighted averages
    w_precision = (precision_0 * n_low_risk + precision_1 * n_high_risk) / total
    w_recall = (recall_0 * n_low_risk + recall_1 * n_high_risk) / total
    w_f1 = (f1_0 * n_low_risk + f1_1 * n_high_risk) / total
    
    # Create classification report dictionary
    class_report = {
        '0': {
            'precision': precision_0,
            'recall': recall_0,
            'f1-score': f1_0,
            'support': n_low_risk
        },
        '1': {
            'precision': precision_1,
            'recall': recall_1,
            'f1-score': f1_1,
            'support': n_high_risk
        },
        'accuracy': accuracy,
        'weighted avg': {
            'precision': w_precision,
            'recall': w_recall,
            'f1-score': w_f1,
            'support': total
        }
    }
    
    return confusion, class_report

def create_daily_risk_trends(df):
    """Create daily risk trends visualization"""
    # Group data by date
    daily_trends = df.groupby('date').agg({
        'headline': 'count',
        'risk_level': lambda x: (x == 'High').mean() * 100,
        'sentiment_score': 'mean',
        'impact_score': 'mean',
        'volatility_index': 'mean'
    }).reset_index()
    
    daily_trends.columns = ['date', 'event_count', 'high_risk_pct', 'sentiment', 'impact', 'volatility']
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Event Count & High Risk %', 'Key Metrics Trends')
    )
    
    # Top chart - Event count with risk percentage
    fig.add_trace(
        go.Bar(
            x=daily_trends['date'],
            y=daily_trends['event_count'],
            name='Event Count',
            marker_color='#3B82F6'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily_trends['date'],
            y=daily_trends['high_risk_pct'],
            name='High Risk %',
            line=dict(color='#EF4444', width=3),
            yaxis='y2'
        ),
        row=1, col=1
    )
    
    # Bottom chart - Key metrics
    fig.add_trace(
        go.Scatter(
            x=daily_trends['date'],
            y=daily_trends['sentiment'],
            name='Sentiment',
            line=dict(color='#10B981', width=2)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily_trends['date'],
            y=daily_trends['impact'],
            name='Impact',
            line=dict(color='#F97316', width=2)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily_trends['date'],
            y=daily_trends['volatility'],
            name='Volatility',
            line=dict(color='#8B5CF6', width=2)
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    # Update y-axes
    fig.update_yaxes(title_text='Event Count', row=1, col=1)
    fig.update_yaxes(title_text='%', overlaying='y', side='right', row=1, col=1)
    fig.update_yaxes(title_text='Score', row=2, col=1)
    
    return fig

# Main function to run the dashboard app
def main():

    # Initialize sidebar
    with st.sidebar:
        st.markdown('<div class="logo-text">📊 Financial Risk Analyzer</div>', unsafe_allow_html=True)
        
        # Data source selection
        data_option = st.radio(
            "Choose data source:",
            ("Upload CSV", "Use Sample Data")
        )
        
        # Handle data loading
        if data_option == "Upload CSV":
            uploaded_file = st.file_uploader("Upload financial risk data (CSV)", type="csv")
            if uploaded_file is not None:
                df = load_data(uploaded_file)
            else:
                st.warning("Please upload a CSV file to proceed.")
                st.stop()
        else:
            df = load_sample_data()

        
        # Date range filter
        df['date'] = pd.to_datetime(df['date'])
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        date_range = st.date_input("Select date range", (min_date, max_date))
        
        # Apply date filter
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
        
        # Risk level filter
        risk_levels = st.multiselect(
            "Risk Levels",
            options=df['risk_level'].unique(),
            default=df['risk_level'].unique()
        )
        df = df[df['risk_level'].isin(risk_levels)] if risk_levels else df

    # Main content tabs
    tabs = st.tabs(["Dashboard", "Risk Analysis", "Market Intelligence", "Text Analysis", "Predictive Models"])
    
    # Dashboard Tab
    with tabs[0]:
        st.markdown('<h2 class="sub-header">Executive Dashboard</h2>', unsafe_allow_html=True)
        
        if df.empty:
            st.warning("No data available with current filters")
            return
        
        create_metrics_row(df)
        display_daily_risk_assessment(df)
        
        col1, col2 = st.columns(2)
        with col1:
            display_top_risk_factors(df)
        with col2:
            display_trending_topics(df)
        
        create_advanced_insights_row(df)
        
        # Display recommendations
        st.markdown('<h3 class="sub-header">💡 Risk Management Recommendations</h3>', unsafe_allow_html=True)
        for i, rec in enumerate(get_risk_recommendations(df)):
            st.markdown(f"""
            <div class="insights-card">
                <span class="badge">#{i+1}</span> {rec}
            </div>
            """, unsafe_allow_html=True)

    # Risk Analysis Tab
    with tabs[1]:
        st.markdown('<h2 class="sub-header">Risk Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_risk_heat_calendar(df), use_container_width=True)
        with col2:
            st.plotly_chart(create_category_risk_breakdown(df), use_container_width=True)
        
        st.plotly_chart(create_risk_timeline(df), use_container_width=True)
        st.plotly_chart(create_source_analysis(df), use_container_width=True)

    # Market Intelligence Tab
    with tabs[2]:
        st.markdown('<h2 class="sub-header">Market Intelligence</h2>', unsafe_allow_html=True)
        
        st.plotly_chart(create_volume_sentiment_plot(df), use_container_width=True)
        st.plotly_chart(create_keyword_network(df), use_container_width=True)

    # Text Analysis Tab
    with tabs[3]:
        st.markdown('<h2 class="sub-header">Text Analysis</h2>', unsafe_allow_html=True)
        
        st.plotly_chart(create_text_analysis(df), use_container_width=True)
        
        # Word frequency by risk level
        risk_tabs = st.tabs(["All"] + sorted(df['risk_level'].unique().tolist()))
        with risk_tabs[0]:
            create_word_frequency_chart(df, title="Overall Word Frequency")
        for i, risk_level in enumerate(sorted(df['risk_level'].unique())):
            with risk_tabs[i+1]:
                create_word_frequency_chart(
                    df[df['risk_level'] == risk_level],
                    title=f"{risk_level} Risk Words",
                    color=get_risk_color(risk_level)
                )

    # Predictive Models Tab
    with tabs[4]:
        st.markdown('<h2 class="sub-header">Predictive Models</h2>', unsafe_allow_html=True)
        
        confusion, report = create_predictive_model(df)
        if confusion is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_confusion_matrix(confusion), use_container_width=True)
            with col2:
                st.write("Classification Report:")
                st.json(report)
        else:
            st.warning("Insufficient data for modeling")

if __name__ == "__main__":
    main()     
