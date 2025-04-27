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

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Set page config
st.set_page_config(
    page_title="Financial Risk Analyzer 📊",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💰"
)

# Custom CSS for better styling - Enhanced with new card styles and animations
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #E2E8F0;
        animation: fadeIn 1.5s;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2563EB;
        padding-top: 1rem;
        margin-top: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: white;
        margin-bottom: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        text-align: center;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: scale(1.05);
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2563EB, #1E40AF);
        color: white;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
        transform: scale(1.05);
    }
    .stProgress .st-eb {
        background-color: #10B981;
    }
    .tab-subheader {
        font-size: 1.2rem;
        font-weight: 600;
        color: #4B5563;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        font-size: 0.8rem;
        color: #6B7280;
        margin-top: 2rem;
        border-top: 1px solid #E5E7EB;
    }
    .sidebar-footer {
        position: fixed;
        bottom: 0;
        padding: 10px;
        background-color: rgba(30, 58, 138, 0.9);
        width: 100%;
        text-align: center;
        color: white;
        font-size: 0.8rem;
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        color: #2563EB;
    }
    .gradient-text {
        background: linear-gradient(90deg, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .animate-in {
        animation: fadeIn 1s ease-out;
    }
    /* Custom tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 120px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    /* Dark/light mode toggle */
    .toggle-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
    }
    .toggle-switch {
        position: relative;
        display: inline-block;
        width: 60px;
        height: 34px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def load_sample_data():
    df = pd.read_csv("Dataset/financial_risk_dataset_enhanced.csv")
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
        <div class="metric-card" style="background-color: #DBEAFE;">
            <h3>📊 Data Points</h3>
            <h2>{}</h2>
        </div>
        """.format(df.shape[0]), unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-card" style="background-color: #E0F2FE;">
            <h3>📈 Risk Categories</h3>
            <h2>{}</h2>
        </div>
        """.format(df['risk_level'].nunique()), unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card" style="background-color: #DBEAFE;">
            <h3>🔍 News Sources</h3>
            <h2>{}</h2>
        </div>
        """.format(df['source'].nunique()), unsafe_allow_html=True)
        
    with col4:
        avg_sentiment = df['sentiment_score'].mean()
        sentiment_emoji = "😊" if avg_sentiment > 0 else "😐" if avg_sentiment == 0 else "😔"
        st.markdown("""
        <div class="metric-card" style="background-color: #E0F2FE;">
            <h3>{} Avg Sentiment</h3>
            <h2>{:.2f}</h2>
        </div>
        """.format(sentiment_emoji, avg_sentiment), unsafe_allow_html=True)

# New function for additional visualizations
def create_trend_analysis(df):
    # Create a date column from the data (if not present, simulate it)
    if 'date' not in df.columns:
        # Simulate dates for demonstration
        date_range = pd.date_range(end=datetime.now(), periods=len(df))
        df['date'] = date_range
    
    # Aggregate data by date
    daily_data = df.groupby(df['date'].dt.date).agg({
        'sentiment_score': 'mean',
        'impact_score': 'mean',
        'risk_level': lambda x: x.value_counts().index[0]  # most common risk level for the day
    }).reset_index()
    
    # Create interactive time series chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add sentiment score line
    fig.add_trace(
        go.Scatter(
            x=daily_data['date'],
            y=daily_data['sentiment_score'],
            name='Sentiment Score',
            line=dict(color='#3B82F6', width=2),
            mode='lines+markers'
        ),
        secondary_y=False
    )
    
    # Add impact score line
    fig.add_trace(
        go.Scatter(
            x=daily_data['date'],
            y=daily_data['impact_score'],
            name='Impact Score',
            line=dict(color='#EF4444', width=2),
            mode='lines+markers'
        ),
        secondary_y=True
    )
    
    # Update layout and axes
    fig.update_layout(
        title='📈 Financial Sentiment & Impact Trends',
        hovermode='x unified',
        height=500
    )
    
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=False)
    fig.update_yaxes(title_text="Impact Score", secondary_y=True)
    
    return fig

# New function for source reliability visualization
def create_source_reliability_chart(df):
    # Calculate source reliability based on sentiment variance and impact
    source_reliability = df.groupby('source').agg({
        'sentiment_score': ['mean', 'std'],
        'impact_score': 'mean',
        'risk_level': lambda x: (x == 'High').mean() * 100  # Percentage of high risk entries
    }).reset_index()
    
    # Flatten multi-level columns
    source_reliability.columns = ['source', 'avg_sentiment', 'sentiment_std', 'avg_impact', 'high_risk_pct']
    
    # Create reliability score
    source_reliability['reliability_score'] = (
        (1 - source_reliability['sentiment_std']) * 0.4 + 
        source_reliability['avg_impact'] * 0.3 + 
        (100 - source_reliability['high_risk_pct']) / 100 * 0.3
    )
    
    # Create bar chart with custom coloring
    fig = px.bar(
        source_reliability.sort_values('reliability_score', ascending=False).head(10),
        x='source',
        y='reliability_score',
        color='reliability_score',
        color_continuous_scale='RdYlGn',
        text=source_reliability['reliability_score'].round(2),
        title='🏆 News Source Reliability Index',
        labels={'reliability_score': 'Reliability Score', 'source': 'Source'}
    )
    
    fig.update_layout(
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig

def main():
    # Header with emojis and animation
    st.markdown('<h1 class="main-header">🚀 Financial Risk Analyzer with GNN 💰</h1>', unsafe_allow_html=True)
    
    # Sidebar with improved styling
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/100/financial-growth-analysis.png", width=100)
        st.markdown("## ⚙️ Configuration")
        
        # Sample data section
        st.markdown("### 📊 Sample Data")
        sample_df = load_sample_data()
        csv_bytes = convert_df_to_csv(sample_df)
        
        st.download_button(
            label="📥 Download Sample Dataset",
            data=csv_bytes,
            file_name="financial_risk_dataset.csv",
            mime="text/csv"
        )
        
        uploaded_file = st.file_uploader("📂 Upload Your Dataset", type=["csv"])
        
        # Model parameters with tooltips
        st.markdown("### 🔧 Model Parameters")
        
        st.markdown("""
        <div class="tooltip">Epochs
          <span class="tooltiptext">Higher values may improve accuracy but take longer to train</span>
        </div>
        """, unsafe_allow_html=True)
        
        epochs = st.slider("🔄 Training Epochs", 10, 300, 200)
        
        st.markdown("""
        <div class="tooltip">Hidden Channels
          <span class="tooltiptext">Controls model complexity</span>
        </div>
        """, unsafe_allow_html=True)
        
        hidden_channels = st.slider("🧠 Hidden Channels", 16, 128, 64)
        
        # Theme switcher (visual only, doesn't actually change theme)
        st.markdown("### 🎨 Appearance")
        theme_mode = st.radio("Theme", ["Light", "Dark"], horizontal=True)
        
        st.markdown("---")
        st.markdown("### 🔍 App Info")
        st.info("This app uses Graph Neural Networks to analyze financial risk based on news headlines and various metrics.")
        
        # Developer attribution in sidebar
        st.markdown("---")
        st.markdown("""
        <div class="sidebar-footer">
            <p>Developed by <span style="font-weight: bold;">Shreyas Kasture</span> ✨</p>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        # Create tabs for better organization - added a new Trends tab
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "🔍 Text Analysis", 
            "📈 Visualizations",
            "📉 Trends",
            "🤖 Model Training"
        ])
        
        with tab1:
            st.markdown('<h2 class="sub-header">📊 Data Overview</h2>', unsafe_allow_html=True)
            
            # Key metrics in attractive cards
            create_metrics_row(df)
            
            # Quick stats summary with improved styling
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Add a quick insights section
            top_risk = df['risk_level'].value_counts().idxmax()
            top_category = df['category'].value_counts().idxmax()
            risk_corr = df['sentiment_score'].corr(df['impact_score']).round(2)
            
            st.markdown(f"""
            <div style="padding: 1rem; background: linear-gradient(90deg, #dbeafe, #e0f2fe); border-radius: 8px; margin-bottom: 1rem;">
                <h3>🔍 Quick Insights</h3>
                <p>• Most common risk level: <strong>{top_risk}</strong></p>
                <p>• Most frequent category: <strong>{top_category}</strong></p>
                <p>• Sentiment-Impact correlation: <strong>{risk_corr}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.checkbox("👀 Show Raw Data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Show column info
            if st.checkbox("📋 Show Column Information"):
                col_desc = {
                    "headline": "Financial news headline text",
                    "category": "News category (e.g., Markets, Economy)",
                    "source": "Publication source",
                    "risk_level": "Assigned risk level",
                    "sentiment_score": "Sentiment analysis score (-1 to 1)",
                    "impact_score": "Estimated financial impact score"
                }
                
                for col, desc in col_desc.items():
                    if col in df.columns:
                        st.markdown(f"**{col}**: {desc}")
            st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            st.markdown('<h2 class="sub-header">🔍 Text Analysis</h2>', unsafe_allow_html=True)
            
            # Text preprocessing card
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="tab-subheader">📝 Text Preprocessing</p>', unsafe_allow_html=True)
            
            df['processed_text'] = df['headline'].apply(preprocess_text)
            
            if st.checkbox("📋 Show Text Processing Examples"):
                for i in range(3):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Original**: {df['headline'].iloc[i]}")
                    with col2:
                        st.markdown(f"**Processed**: {df['processed_text'].iloc[i]}")
                    st.markdown("---")
            st.markdown('</div>', unsafe_allow_html=True)

            # TF-IDF Processing - Enhanced visualization
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="tab-subheader">🔤 TF-IDF Analysis</p>', unsafe_allow_html=True)
            
            tfidf_vectorizer = TfidfVectorizer(max_features=1000)
            tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])
            
            # Word cloud alternative - show top terms with improved visualization
            tfidf_sum = pd.DataFrame(
                tfidf_matrix.sum(axis=0).T,
                index=tfidf_vectorizer.get_feature_names_out(),
                columns=['sum']
            ).sort_values('sum', ascending=False)
            
            top_terms = tfidf_sum.head(15)
            
            # Choose display type for better visualization
            viz_type = st.radio("Select visualization type:", ["Bar Chart", "Word Cloud"], horizontal=True)
            
            if viz_type == "Bar Chart":
                fig = px.bar(
                    top_terms, 
                    y=top_terms.index, 
                    x='sum',
                    orientation='h',
                    title='📊 Top 15 Important Terms',
                    color='sum',
                    color_continuous_scale='Blues',
                    text_auto='.2s'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Simple visual representation of word importance
                st.markdown("<h3>📝 Key Terms Importance</h3>", unsafe_allow_html=True)
                
                for term, value in top_terms.iterrows():
                    size = int(50 + (value['sum'] / top_terms['sum'].max()) * 150)
                    opacity = 0.3 + (value['sum'] / top_terms['sum'].max()) * 0.7
                    st.markdown(f"""
                    <span style="font-size: {size}%; opacity: {opacity}; margin: 5px; display: inline-block; color: #3B82F6;">
                        {term}
                    </span>
                    """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with tab3:
            st.markdown('<h2 class="sub-header">📈 Visualizations</h2>', unsafe_allow_html=True)
            
            viz_option = st.selectbox(
                "🎨 Choose visualization",
                ["📊 Category Distribution", "⚠️ Risk Level Analysis", "😊 Sentiment Analysis", "🔄 Category Relationships", "🏆 Source Reliability"]
            )

            st.markdown('<div class="card">', unsafe_allow_html=True)
            if viz_option == "📊 Category Distribution":
                # Enhanced category distribution with Plotly
                category_counts = df['category'].value_counts().reset_index()
                category_counts.columns = ['category', 'count']
                
                # Add percentage column
                category_counts['percentage'] = (category_counts['count'] / category_counts['count'].sum() * 100).round(1)
                
                # Let user switch between chart types
                chart_type = st.radio("Select chart type:", ["Bar", "Pie", "Treemap"], horizontal=True)
                
                if chart_type == "Bar":
                    fig = px.bar(
                        category_counts,
                        x='category',
                        y='count',
                        color='count',
                        color_continuous_scale='Viridis',
                        title='📊 News Category Distribution',
                        labels={'count': 'Number of Articles', 'category': 'Category'},
                        text=category_counts['percentage'].apply(lambda x: f"{x}%"),
                        hover_data=['count', 'percentage']
                    )
                elif chart_type == "Pie":
                    fig = px.pie(
                        category_counts,
                        values='count',
                        names='category',
                        title='📊 News Category Distribution',
                        hole=0.4,
                        hover_data=['percentage'],
                        labels={'percentage': 'Percentage (%)'}
                    )
                else:  # Treemap
                    fig = px.treemap(
                        category_counts,
                        path=['category'],
                        values='count',
                        title='📊 News Category Distribution',
                        color='count',
                        color_continuous_scale='Viridis',
                        hover_data=['percentage']
                    )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_option == "⚠️ Risk Level Analysis":
                # Enhanced risk level distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk level distribution
                    risk_counts = df['risk_level'].value_counts().reset_index()
                    risk_counts.columns = ['risk_level', 'count']
                    risk_counts['percentage'] = (risk_counts['count'] / risk_counts['count'].sum() * 100).round(1)
                    
                    # Custom color map based on risk level
                    color_map = {'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#EF4444'}
                    colors = risk_counts['risk_level'].map(color_map).tolist()
                    
                    fig = px.pie(
                        risk_counts,
                        values='count',
                        names='risk_level',
                        title='⚠️ Risk Level Distribution',
                        hole=0.4,
                        hover_data=['percentage'],
                        color_discrete_sequence=colors if all(level in color_map for level in risk_counts['risk_level']) else px.colors.sequential.RdBu,
                    )
                    fig.update_traces(textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Risk by category heatmap - enhanced
                    risk_category = pd.crosstab(df['category'], df['risk_level'])
                    
                    # Normalize to percentages for better insight
                    risk_category_pct = risk_category.div(risk_category.sum(axis=1), axis=0) * 100
                    
                    view_mode = st.radio("View as:", ["Counts", "Percentages"], horizontal=True)
                    
                    display_data = risk_category if view_mode == "Counts" else risk_category_pct
                    
                    fig = px.imshow(
                        display_data,
                        text_auto=True if view_mode == "Counts" else '.1f',
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title='🔥 Risk Level by Category',
                        labels={"color": "Count" if view_mode == "Counts" else "Percentage (%)"}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
            elif viz_option == "😊 Sentiment Analysis":
                # Enhanced sentiment visualizations
                # Add a filter option
                category_filter = st.multiselect(
                    "Filter by category:",
                    options=df['category'].unique(),
                    default=df['category'].unique()
                )
                
                filtered_df = df[df['category'].isin(category_filter)]
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("😊 Sentiment by Category", "💥 Impact by Category")
                )
                
                # Sentiment by category
                for category in filtered_df['category'].unique():
                    category_data = filtered_df[filtered_df['category'] == category]
                    
                    fig.add_trace(
                        go.Box(
                            y=category_data['sentiment_score'],
                            name=category,
                            boxmean=True
                        ),
                        row=1, col=1
                    )
                
                # Impact by category
                for category in filtered_df['category'].unique():
                    category_data = filtered_df[filtered_df['category'] == category]
                    
                    fig.add_trace(
                        go.Box(
                            y=category_data['impact_score'],
                            name=category,
                            boxmean=True
                        ),
                        row=1, col=2
                    )
                
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Sentiment vs Impact scatter - enhanced with hover info
                scatter_fig = px.scatter(
                    filtered_df,
                    x='sentiment_score',
                    y='impact_score',
                    color='risk_level',
                    size='impact_score',
                    hover_data=['headline', 'source'],
                    title='🔄 Sentiment vs Impact Analysis',
                    labels={
                        'sentiment_score': 'Sentiment Score',
                        'impact_score': 'Impact Score',
                        'risk_level': 'Risk Level'
                    },
                    color_discrete_map={'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#EF4444'},
                    size_max=15,
                    opacity=0.7
                )
                
                # Add quadrant lines to help interpret
                scatter_fig.add_shape(
                    type="line", line=dict(dash="dash", width=1, color="gray"),
                    x0=0, y0=filtered_df['impact_score'].min(), 
                    x1=0, y1=filtered_df['impact_score'].max()
                )
                scatter_fig.add_shape(
                    type="line", line=dict(dash="dash", width=1, color="gray"),
                    x0=filtered_df['sentiment_score'].min(), y0=filtered_df['impact_score'].mean(),
                    x1=filtered_df['sentiment_score'].max(), y1=filtered_df['impact_score'].mean()
                )
                
                # Add quadrant annotations
                scatter_fig.add_annotation(
                    x=0.75, y=filtered_df['impact_score'].max() * 0.9,
                    text="High Impact, Positive Sentiment",
                    showarrow=False, font=dict(size=10, color="#10B981")
                )
                scatter_fig.add_annotation(
                    x=-0.75, y=filtered_df['impact_score'].max() * 0.9,
                    text="High Impact, Negative Sentiment",
                    showarrow=False, font=dict(size=10, color="#EF4444")
                )
                
                scatter_fig.update_layout(height=500)
                st.plotly_chart(scatter_fig, use_container_width=True)
                
            elif viz_option == "🔄 Category Relationships":
                # Enhanced network visualization
                st.markdown("### 🔄 Category Relationship Network")
                st.info("This visualization shows how different categories are connected based on similar risk profiles.")
                
                # Create category relationships based on risk levels
                category_matrix = pd.crosstab(df['category'], df['risk_level'])
                
                # Normalize to get profiles
                category_profiles = category_matrix.div(category_matrix.sum(axis=1), axis=0)
                
                # Visualization options
                viz_type = st.radio("Select visualization:", ["Heatmap", "3D Network"], horizontal=True)
                
                if viz_type == "Heatmap":
                    # Visualize as a heatmap with improved color scheme
                    fig = px.imshow(
                        category_profiles,
                        text_auto='.2f',
                        aspect="auto",
                        color_continuous_scale='Viridis',
                        title='🔄 Category Risk Profiles'
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Create 3D visualization of category relationships
                    # Use PCA for dimensionality reduction
                    from sklearn.decomposition import PCA
                    
                    # Prepare the data
                    X = category_profiles.values
                    
                    # Apply PCA to get coordinates for visualization
                    pca = PCA(n_components=3)
                    coords = pca.fit_transform(X)
                    
                    # Create a DataFrame for plotting
                    network_df = pd.DataFrame({
                        'category': category_profiles.index,
                        'x': coords[:, 0],
                        'y': coords[:, 1],
                        'z': coords[:, 2],
                        'size': category_matrix.sum(axis=1) / category_matrix.sum(axis=1).max() * 20 + 10
                    })
                    
                    # Create 3D scatter plot
                    fig = px.scatter_3d(
                        network_df,
                        x='x', y='y', z='z',
                        color='category',
                        size='size',
                        hover_name='category',
                        title='🔄 3D Category Network Map'
                    )
                    
# Update layout for 3D visualization
                    fig.update_layout(
                        height=700,
                        scene=dict(
                            xaxis_title='Component 1',
                            yaxis_title='Component 2',
                            zaxis_title='Component 3'
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add some insights about the network visualization
                    st.markdown("""
                    <div style="background-color: #f8fafc; padding: 15px; border-radius: 10px; border-left: 5px solid #3B82F6;">
                        <h4>📊 Network Insights</h4>
                        <p>Categories that appear closer together in the 3D space have similar risk profiles.
                        The size of each node represents the relative number of articles in that category.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            elif viz_option == "🏆 Source Reliability":
                # Source reliability visualization
                st.markdown("### 🏆 Source Reliability Analysis")
                st.info("This visualization shows how reliable different news sources are based on sentiment consistency, impact scores, and risk levels.")
                
                # Create and display source reliability chart
                reliability_chart = create_source_reliability_chart(df)
                st.plotly_chart(reliability_chart, use_container_width=True)
                
                # Add detailed explanation
                st.markdown("""
                <div style="background-color: #f0f9ff; padding: 15px; border-radius: 10px; border-left: 5px solid #0ea5e9;">
                    <h4>ℹ️ How Reliability is Calculated</h4>
                    <p>The reliability score is a composite metric that factors in:</p>
                    <ul>
                        <li><strong>Sentiment consistency</strong>: Lower variance in sentiment indicates more consistent reporting</li>
                        <li><strong>Impact accuracy</strong>: Higher average impact scores suggest the source reports on significant events</li>
                        <li><strong>Risk assessment</strong>: Sources with balanced risk reporting rather than sensationalized high-risk content</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        with tab4:
            st.markdown('<h2 class="sub-header">📉 Trend Analysis</h2>', unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # Create trends visualization
            trends_fig = create_trend_analysis(df)
            st.plotly_chart(trends_fig, use_container_width=True)
            
            # Add time period filter with custom date slider
            col1, col2 = st.columns(2)
            with col1:
                # Get date range
                if 'date' in df.columns:
                    min_date = df['date'].min().date()
                    max_date = df['date'].max().date()
                    
                    # Create date range slider
                    date_range = st.date_input(
                        "Select date range:",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date
                    )
            
            with col2:
                # Add granularity selector
                granularity = st.selectbox(
                    "Select time aggregation:",
                    options=["Daily", "Weekly", "Monthly"]
                )
            
            # Category trend analysis
            st.markdown("### 📊 Category Trend Analysis")
            
            # Select categories to show
            selected_categories = st.multiselect(
                "Select categories to analyze:",
                options=df['category'].unique(),
                default=df['category'].unique()[:3]  # Default to first 3 categories
            )
            
            if selected_categories:
                # Filter data for selected categories
                category_data = df[df['category'].isin(selected_categories)]
                
                # Get date column if exists, otherwise create one
                if 'date' not in category_data.columns:
                    date_range = pd.date_range(end=datetime.now(), periods=len(df))
                    category_data['date'] = date_range
                
                # Group by category and date
                if granularity == "Daily":
                    category_trends = category_data.groupby(['category', category_data['date'].dt.date]).agg({
                        'sentiment_score': 'mean',
                        'impact_score': 'mean'
                    }).reset_index()
                elif granularity == "Weekly":
                    category_data['week'] = category_data['date'].dt.isocalendar().week
                    category_data['year'] = category_data['date'].dt.isocalendar().year
                    category_trends = category_data.groupby(['category', 'year', 'week']).agg({
                        'sentiment_score': 'mean',
                        'impact_score': 'mean',
                        'date': 'first'  # Keep a date for plotting
                    }).reset_index()
                else:  # Monthly
                    category_data['month'] = category_data['date'].dt.month
                    category_data['year'] = category_data['date'].dt.year
                    category_trends = category_data.groupby(['category', 'year', 'month']).agg({
                        'sentiment_score': 'mean',
                        'impact_score': 'mean',
                        'date': 'first'  # Keep a date for plotting
                    }).reset_index()
                
                # Create line chart
                fig = px.line(
                    category_trends,
                    x='date',
                    y='sentiment_score',
                    color='category',
                    title=f'📈 {granularity} Sentiment Trends by Category',
                    labels={'sentiment_score': 'Average Sentiment', 'date': 'Date'},
                    markers=True
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add impact score chart
                fig2 = px.line(
                    category_trends,
                    x='date',
                    y='impact_score',
                    color='category',
                    title=f'💥 {granularity} Impact Trends by Category',
                    labels={'impact_score': 'Average Impact', 'date': 'Date'},
                    markers=True
                )
                
                fig2.update_layout(height=500)
                st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        with tab5:
            st.markdown('<h2 class="sub-header">🤖 GNN Model Training</h2>', unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📊 Graph Neural Network for Risk Classification")
            st.info("This model uses a Graph Convolutional Network (GCN) to classify financial news into risk categories, considering both textual features and relationships between news items.")
            
            # Create feature vector
            st.markdown("#### 1️⃣ Feature Extraction")
            
            # Extract TF-IDF features and perform dimensionality reduction
            with st.spinner("Extracting features..."):
                tfidf_vectorizer = TfidfVectorizer(max_features=500)
                tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])
                
                # Dimensionality reduction
                svd = TruncatedSVD(n_components=50)
                text_features = svd.fit_transform(tfidf_matrix)
                
                # Combine with numerical features
                features = np.hstack((
                    text_features,
                    df[['sentiment_score', 'impact_score']].values
                ))
                
                # Display feature extraction progress
                st.success(f"✅ Extracted {features.shape[1]} features from {features.shape[0]} data points")
                
                # Progress bar for visual feedback
                st.progress(100)
            
            # Create graph structure
            st.markdown("#### 2️⃣ Graph Construction")
            
            with st.spinner("Constructing graph..."):
                # Create edges based on similarity
                from sklearn.metrics.pairwise import cosine_similarity
                
                # Sample a smaller subset if dataset is large for demonstration
                max_nodes = 1000  # Adjust based on performance
                if features.shape[0] > max_nodes:
                    # Sample data for visualization
                    indices = np.random.choice(features.shape[0], max_nodes, replace=False)
                    sample_features = features[indices]
                    sample_labels = df['risk_level'].iloc[indices]
                else:
                    sample_features = features
                    sample_labels = df['risk_level']
                
                # Compute similarity
                similarity = cosine_similarity(sample_features)
                
                # Keep only strong connections (above threshold)
                threshold = 0.7
                similarity[similarity < threshold] = 0
                
                # Create edge list
                edges = []
                for i in range(similarity.shape[0]):
                    for j in range(i+1, similarity.shape[0]):
                        if similarity[i, j] > 0:
                            edges.append((i, j))
                            edges.append((j, i))  # Add both directions for undirected graph
                
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                
                # Create node features
                x = torch.tensor(sample_features, dtype=torch.float)
                
                # Create labels
                # Encode risk levels to numeric
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(sample_labels)
                y = torch.tensor(y, dtype=torch.long)
                
                # Create graph data
                graph_data = Data(x=x, edge_index=edge_index, y=y)
                
                # Display graph construction progress
                st.success(f"✅ Constructed graph with {x.shape[0]} nodes and {edge_index.shape[1]} edges")
                
                # Progress bar visualization
                st.progress(100)
            
            # Define the GNN model architecture
            st.markdown("#### 3️⃣ GNN Model Architecture")
            
            # Simple GCN architecture
            class GCN(torch.nn.Module):
                def __init__(self, num_features, hidden_channels, num_classes):
                    super(GCN, self).__init__()
                    torch.manual_seed(42)
                    self.conv1 = GCNConv(num_features, hidden_channels)
                    self.conv2 = GCNConv(hidden_channels, hidden_channels)
                    self.linear = torch.nn.Linear(hidden_channels, num_classes)
                
                def forward(self, x, edge_index):
                    x = self.conv1(x, edge_index)
                    x = x.relu()
                    x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
                    x = self.conv2(x, edge_index)
                    x = x.relu()
                    x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
                    x = self.linear(x)
                    return x
            
            # Display model architecture
            st.markdown("""
            ```
            GCN(
              (conv1): GCNConv(in_channels={}, out_channels={})
              (conv2): GCNConv(in_channels={}, out_channels={})
              (linear): Linear(in_features={}, out_features={})
            )
            ```
            """.format(
                features.shape[1], hidden_channels,
                hidden_channels, hidden_channels,
                hidden_channels, len(label_encoder.classes_)
            ))
            
            # Model training section
            st.markdown("#### 4️⃣ Model Training")
            
            # Add train/test split for evaluation
            train_mask, test_mask = train_test_split(
                np.arange(len(y)),
                test_size=0.2,
                stratify=y.numpy(),
                random_state=42
            )
            
            train_mask = torch.tensor(train_mask, dtype=torch.long)
            test_mask = torch.tensor(test_mask, dtype=torch.long)
            
            # Initialize model
            model = GCN(
                num_features=features.shape[1],
                hidden_channels=hidden_channels,
                num_classes=len(label_encoder.classes_)
            )
            
            # Training parameters
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            criterion = torch.nn.CrossEntropyLoss()
            
            # Display training parameters
            st.markdown("""
            **Training Parameters:**
            - Optimizer: Adam
            - Learning Rate: 0.01
            - Weight Decay: 0.0005
            - Loss Function: Cross Entropy
            - Epochs: {}
            """.format(epochs))
            
            # Train button
            if st.button("🚀 Train Model", type="primary"):
                # Training loop with progress bar
                progress_bar = st.progress(0)
                train_losses = []
                test_accuracies = []
                
                # Placeholder for the chart
                chart_placeholder = st.empty()
                
                # Training loop
                model.train()
                for epoch in range(epochs):
                    # Forward pass
                    optimizer.zero_grad()
                    out = model(x, edge_index)
                    loss = criterion(out[train_mask], y[train_mask])
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Compute test accuracy
                    model.eval()
                    with torch.no_grad():
                        out = model(x, edge_index)
                        pred = out.argmax(dim=1)
                        test_correct = pred[test_mask] == y[test_mask]
                        test_acc = int(test_correct.sum()) / len(test_mask)
                    
                    # Store metrics
                    train_losses.append(loss.item())
                    test_accuracies.append(test_acc)
                    
                    # Update progress bar
                    progress_bar.progress((epoch + 1) / epochs)
                    
                    # Update training chart every 10 epochs
                    if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
                        metrics_df = pd.DataFrame({
                            'Epoch': range(1, len(train_losses) + 1),
                            'Train Loss': train_losses,
                            'Test Accuracy': test_accuracies
                        })
                        
                        # Create dual-axis chart
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        # Add loss line
                        fig.add_trace(
                            go.Scatter(
                                x=metrics_df['Epoch'],
                                y=metrics_df['Train Loss'],
                                name='Training Loss',
                                line=dict(color='#EF4444', width=2)
                            ),
                            secondary_y=False
                        )
                        
                        # Add accuracy line
                        fig.add_trace(
                            go.Scatter(
                                x=metrics_df['Epoch'],
                                y=metrics_df['Test Accuracy'],
                                name='Test Accuracy',
                                line=dict(color='#10B981', width=2)
                            ),
                            secondary_y=True
                        )
                        
                        # Update layout
                        fig.update_layout(
                            title='📈 Training Progress',
                            xaxis_title='Epoch',
                            height=400
                        )
                        
                        fig.update_yaxes(title_text="Training Loss", secondary_y=False)
                        fig.update_yaxes(title_text="Test Accuracy", secondary_y=True)
                        
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Final evaluation
                model.eval()
                with torch.no_grad():
                    out = model(x, edge_index)
                    pred = out.argmax(dim=1)
                    
                    # Compute confusion matrix
                    y_true = y[test_mask].numpy()
                    y_pred = pred[test_mask].numpy()
                    
                    conf_matrix = confusion_matrix(y_true, y_pred)
                    
                    # Create confusion matrix visualization
                    cm_fig = px.imshow(
                        conf_matrix,
                        labels=dict(x="Predicted", y="True"),
                        x=label_encoder.classes_,
                        y=label_encoder.classes_,
                        text_auto=True,
                        color_continuous_scale="Blues",
                        title="Confusion Matrix"
                    )
                    
                    st.plotly_chart(cm_fig, use_container_width=True)
                    
                    # Display classification report
                    report = classification_report(
                        y_true,
                        y_pred,
                        target_names=label_encoder.classes_,
                        output_dict=True
                    )
                    
                    # Convert to DataFrame for better display
                    report_df = pd.DataFrame(report).T
                    st.dataframe(report_df.style.format({
                        'precision': '{:.2f}',
                        'recall': '{:.2f}',
                        'f1-score': '{:.2f}',
                        'support': '{:.0f}'
                    }), use_container_width=True)
                    
                    # Final accuracy message
                    test_acc = int((pred[test_mask] == y[test_mask]).sum()) / len(test_mask)
                    st.success(f"🎉 Model training complete! Final test accuracy: {test_acc:.2%}")
            
            # Show node embeddings visualization
            if st.checkbox("👁️ Show Node Embeddings Visualization"):
                # Create a simplified model for visualization
                vis_model = GCN(features.shape[1], hidden_channels, len(label_encoder.classes_))
                
                # Get embeddings from the GCN's first layer
                with torch.no_grad():
                    node_embeddings = vis_model.conv1(x, edge_index).relu().numpy()
                
                # Reduce to 2D for visualization
                tsne = TSNE(n_components=2, random_state=42)
                embedding_2d = tsne.fit_transform(node_embeddings)
                
                # Prepare for plotting
                embedding_df = pd.DataFrame({
                    'x': embedding_2d[:, 0],
                    'y': embedding_2d[:, 1],
                    'Risk Level': [label_encoder.classes_[i] for i in y]
                })
                
                # Color map
                color_map = {'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#EF4444'}
                
                # Plot embeddings
                fig = px.scatter(
                    embedding_df,
                    x='x',
                    y='y',
                    color='Risk Level',
                    color_discrete_map=color_map if all(level in color_map for level in embedding_df['Risk Level'].unique()) else None,
                    title='Node Embeddings (2D t-SNE Projection)',
                    opacity=0.7
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Display welcome message when no data is loaded
        st.markdown("""
        <div class="card animate-in" style="padding: 2rem; text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📊💸</div>
            <h2>Welcome to the Financial Risk Analyzer!</h2>
            <p>This application uses advanced Graph Neural Networks to analyze and categorize financial news based on risk levels.</p>
            <p>To get started, please upload a CSV file with financial news data or use the sample dataset from the sidebar.</p>
            
            <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
                <div style="text-align: center; max-width: 200px; padding: 1rem; background-color: #eff6ff; border-radius: 8px;">
                    <div class="feature-icon">📝</div>
                    <h3>Text Analysis</h3>
                    <p>Process and analyze news headlines using NLP techniques</p>
                </div>
                
                <div style="text-align: center; max-width: 200px; padding: 1rem; background-color: #f0fdf4; border-radius: 8px;">
                    <div class="feature-icon">📊</div>
                    <h3>Visualizations</h3>
                    <p>Interactive charts and graphs to explore the data</p>
                </div>
                
                <div style="text-align: center; max-width: 200px; padding: 1rem; background-color: #eff6ff; border-radius: 8px;">
                    <div class="feature-icon">🤖</div>
                    <h3>GNN Model</h3>
                    <p>Advanced Graph Neural Network for risk classification</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample screenshots
        st.markdown("""
        <div class="card" style="padding: 1.5rem; margin-top: 1rem;">
            <h3 class="sub-header">📷 Sample Visualizations</h3>
            <p>Here's a preview of what you can do with this app:</p>
            
            <div style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center; margin-top: 1rem;">
                <div style="text-align: center; max-width: 300px;">
                    <img src="https://img.icons8.com/color/96/financial-analytics.png" style="border-radius: 8px; max-width: 100%; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <p>Risk Level Distribution</p>
                </div>
                
                <div style="text-align: center; max-width: 300px;">
                    <img src="https://img.icons8.com/color/96/combo-chart--v1.png" style="border-radius: 8px; max-width: 100%; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <p>Sentiment Analysis</p>
                </div>
                
                <div style="text-align: center; max-width: 300px;">
                    <img src="https://img.icons8.com/color/96/network.png" style="border-radius: 8px; max-width: 100%; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <p>Network Visualization</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Custom footer
    st.markdown("""
    <div class="footer">
        <p>Financial Risk Analyzer with GNN | Created by Shreyas Kasture | © 2025</p>
        <p>Built with Streamlit, PyTorch Geometric, and Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
