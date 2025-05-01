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

# Custom CSS for better styling
st.markdown("""
<style>
    /* Header Styling */
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #E2E8F0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
        text-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2563EB;
        padding-top: 1rem;
        margin-top: 1rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    /* Card Styling */
    .card {
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        background-color: #FFFFFF;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    .footer:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }

    /* Metric Card Styling */
    .metric-card {
        text-align: center;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        background-color: #F0F9FF;
        margin-bottom: 1.2rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background-image: linear-gradient(145deg, #2563EB, #1E40AF);
        color: white;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }
    .css-1d391kg:hover {
        background-color: #1D4ED8;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 30px;
        padding: 0.7rem 1.4rem;
        font-weight: bold;
        transition: background-color 0.3s ease, transform 0.2s ease;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    /* Progress Bar Styling */
    .stProgress > div > div > div {
        background-color: #10B981; /* Smooth green for progress */
        border-radius: 8px;
        height: 12px;
    }

    /* Tab Header Styling */
    .tab-subheader {
        font-size: 1.4rem;
        font-weight: 600;
        color: #4B5563;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E2E8F0;
    }

    /* Footer Styling */
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #7f8c8d;
        font-size: 0.9rem;
        padding-top: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    /* Custom Scrollbar for File List */
    .file-list {
        max-height: 200px;
        overflow-y: auto;
        margin-top: 10px;
        padding: 10px;
        background-color: #F8FAFC;
        border-radius: 8px;
    }
    .footer {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
        margin-top: 50px;
    }
    .file-item {
        display: flex;
        justify-content: space-between;
        padding: 12px;
        margin-bottom: 8px;
        background-color: #FFFFFF;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .file-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* Remove Button Styling in File List */
    .remove-btn {
        color: #DC2626;
        cursor: pointer;
        font-weight: bold;
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

def main():
    # Header with emojis
    st.markdown('<h1 class="main-header">🚀 Financial Risk Analyzer with GNN 💰</h1>', unsafe_allow_html=True)
    
    # Sidebar with improved styling
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        sample_df = load_sample_data()
        csv_bytes = convert_df_to_csv(sample_df)
        
        st.download_button(
            label="📥 Download Dataset",
            data=csv_bytes,
            file_name="financial_risk_dataset.csv",
            mime="text/csv"
        )
        
        uploaded_file = st.file_uploader("📂 Upload Your Dataset", type=["csv"])
        
        st.markdown("### 🔧 Model Parameters")
        epochs = st.slider("🔄 Training Epochs", 10, 300, 200)
        hidden_channels = st.slider("🧠 Hidden Channels", 16, 128, 64)
        
        st.markdown("---")
        st.markdown("### 🔍 App Info")
        st.info("This app uses Graph Neural Networks to analyze financial risk based on news headlines and various metrics.")
        st.markdown("---")
        st.markdown("Developed By Shreyas Kasture")

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        # Create tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", 
            "🔍 Text Analysis", 
            "📈 Visualizations", 
            "🤖 Model Training"
        ])
        
        with tab1:
            st.markdown('<h2 class="sub-header">📊 Data Overview</h2>', unsafe_allow_html=True)
            
            # Key metrics in attractive cards
            create_metrics_row(df)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
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

            # TF-IDF Processing
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="tab-subheader">🔤 TF-IDF Analysis</p>', unsafe_allow_html=True)
            
            tfidf_vectorizer = TfidfVectorizer(max_features=1000)
            tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])
            
            # Word cloud alternative - show top terms
            tfidf_sum = pd.DataFrame(
                tfidf_matrix.sum(axis=0).T,
                index=tfidf_vectorizer.get_feature_names_out(),
                columns=['sum']
            ).sort_values('sum', ascending=False)
            
            top_terms = tfidf_sum.head(15)
            
            fig = px.bar(
                top_terms, 
                y=top_terms.index, 
                x='sum',
                orientation='h',
                title='📊 Top 15 Important Terms',
                color='sum',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with tab3:
            st.markdown('<h2 class="sub-header">📈 Visualizations</h2>', unsafe_allow_html=True)
            
            viz_option = st.selectbox(
                "🎨 Choose visualization",
                ["📊 Category Distribution", "⚠️ Risk Level Analysis", "😊 Sentiment Analysis", "🔄 Category Relationships"]
            )

            st.markdown('<div class="card">', unsafe_allow_html=True)
            if viz_option == "📊 Category Distribution":
                # Enhanced category distribution with Plotly
                category_counts = df['category'].value_counts().reset_index()
                category_counts.columns = ['category', 'count']
                
                fig = px.bar(
                    category_counts,
                    x='category',
                    y='count',
                    color='count',
                    color_continuous_scale='Viridis',
                    title='📊 News Category Distribution',
                    labels={'count': 'Number of Articles', 'category': 'Category'}
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
                    
                    fig = px.pie(
                        risk_counts,
                        values='count',
                        names='risk_level',
                        title='⚠️ Risk Level Distribution',
                        color_discrete_sequence=px.colors.sequential.RdBu,
                        hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Risk by category heatmap
                    risk_category = pd.crosstab(df['category'], df['risk_level'])
                    
                    fig = px.imshow(
                        risk_category,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title='🔥 Risk Level by Category'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
            elif viz_option == "😊 Sentiment Analysis":
                # Enhanced sentiment visualizations
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("😊 Sentiment by Category", "💥 Impact by Category")
                )
                
                # Sentiment by category
                for category in df['category'].unique():
                    category_data = df[df['category'] == category]
                    
                    fig.add_trace(
                        go.Box(
                            y=category_data['sentiment_score'],
                            name=category,
                            boxmean=True
                        ),
                        row=1, col=1
                    )
                
                # Impact by category
                for category in df['category'].unique():
                    category_data = df[df['category'] == category]
                    
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
                
                # Sentiment vs Impact scatter
                scatter_fig = px.scatter(
                    df,
                    x='sentiment_score',
                    y='impact_score',
                    color='risk_level',
                    size='impact_score',
                    hover_data=['headline'],
                    title='🔄 Sentiment vs Impact Analysis',
                    labels={
                        'sentiment_score': 'Sentiment Score',
                        'impact_score': 'Impact Score',
                        'risk_level': 'Risk Level'
                    }
                )
                scatter_fig.update_layout(height=500)
                st.plotly_chart(scatter_fig, use_container_width=True)
                
            elif viz_option == "🔄 Category Relationships":
                # Network graph showing relationships between categories
                st.markdown("### 🔄 Category Relationship Network")
                st.info("This visualization shows how different categories are connected based on similar risk profiles.")
                
                # Create category relationships based on risk levels
                category_matrix = pd.crosstab(df['category'], df['risk_level'])
                
                # Normalize to get profiles
                category_profiles = category_matrix.div(category_matrix.sum(axis=1), axis=0)
                
                # Visualize as a heatmap
                fig = px.imshow(
                    category_profiles,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='Viridis',
                    title='🔄 Category Risk Profiles'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        with tab4:
            st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="tab-subheader">⚙️ GNN Model Configuration</p>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Data Points", df.shape[0])
            with col2:
                st.metric("🧠 Hidden Channels", hidden_channels)
            with col3:
                st.metric("🔄 Training Epochs", epochs)
                
            st.markdown('</div>', unsafe_allow_html=True)
            
            # GNN Model Training
            if st.button("🚀 Train GNN Model"):
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("## 📈 Training Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Prepare graph data
                label_encoders = {}
                for col in ['category', 'source', 'risk_level']:
                    le = LabelEncoder()
                    df[f'{col}_encoded'] = le.fit_transform(df[col])
                    label_encoders[col] = le

                # Create edges
                edges = []
                category_groups = df.groupby('category_encoded')
                for _, group in category_groups:
                    indices = group.index.tolist()
                    if len(indices) > 1:
                        edges.append((indices[0], indices[1]))  # Connect first two nodes per category

                edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long) if edges else torch.empty((2, 0), dtype=torch.long)
                                
                for category in df['category_encoded'].unique():
                    nodes = df[df['category_encoded'] == category].index.tolist()
                    for i in range(len(nodes)):
                        for j in range(i+1, len(nodes)):
                            edges.append((nodes[i], nodes[j]))
                
                edge_index = torch.tensor(list(zip(*[(u, v) for u, v in edges])), dtype=torch.long)

                # Node features
                node_features = np.hstack((
                    tfidf_matrix.toarray(),
                    df[['sentiment_score', 'impact_score', 'category_encoded', 'source_encoded']].values
                ))
                x = torch.tensor(node_features, dtype=torch.float)
                y = torch.tensor(df['risk_level_encoded'].values, dtype=torch.long)

                # Create data object
                data = Data(x=x, edge_index=edge_index, y=y)

                # Add train/test split
                indices = list(range(len(df)))
                train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
                
                train_mask = torch.zeros(len(df), dtype=torch.bool)
                test_mask = torch.zeros(len(df), dtype=torch.bool)
                train_mask[train_indices] = True
                test_mask[test_indices] = True

                # Initialize model
                class GCN(torch.nn.Module):
                    def __init__(self, num_features, hidden_channels, num_classes):
                        super().__init__()
                        self.conv1 = GCNConv(num_features, hidden_channels)
                        self.dropout = torch.nn.Dropout(0.5)
                        self.lin = torch.nn.Linear(hidden_channels, num_classes)

                    def forward(self, x, edge_index):
                        x = self.conv1(x, edge_index).relu()
                        x = self.dropout(x)
                        return self.lin(x)

                model = GCN(num_features=data.num_node_features,
                          hidden_channels=hidden_channels,
                          num_classes=len(df['risk_level'].unique()))

                # Training setup
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = torch.nn.CrossEntropyLoss()

                # Training loop
                train_accuracies = []
                test_accuracies = []
                losses = []
                
                for epoch in range(epochs):
                    model.train()
                    optimizer.zero_grad()
                    out = model(data.x, data.edge_index)
                    loss = criterion(out[train_mask], data.y[train_mask])
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

                    # Calculate accuracies
                    with torch.no_grad():
                        model.eval()
                        pred = out.argmax(dim=1)
                        train_acc = (pred[train_mask] == data.y[train_mask]).sum().item() / train_mask.sum().item()
                        test_acc = (pred[test_mask] == data.y[test_mask]).sum().item() / test_mask.sum().item()
                        
                    train_accuracies.append(train_acc)
                    test_accuracies.append(test_acc)

                    # Update progress
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    
                    # Emoji based on test accuracy trend
                    accuracy_emoji = "🚀" if len(test_accuracies) > 1 and test_accuracies[-1] > test_accuracies[-2] else "📊"
                    
                    status_text.text(f"{accuracy_emoji} Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} | Train Acc: {train_acc:.2f} | Test Acc: {test_acc:.2f}")

                st.success("✅ Training completed successfully!")
                
                # Show accuracy graph with Plotly
                st.subheader("📈 Training Progress")
                
                # Create DataFrame for Plotly
                progress_df = pd.DataFrame({
                    'Epoch': list(range(1, epochs + 1)),
                    'Train Accuracy': train_accuracies,
                    'Test Accuracy': test_accuracies,
                    'Loss': losses
                })
                
                # Plot accuracies
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Scatter(
                        x=progress_df['Epoch'],
                        y=progress_df['Train Accuracy'],
                        name='Train Accuracy',
                        line=dict(color='#3B82F6', width=2)
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=progress_df['Epoch'],
                        y=progress_df['Test Accuracy'],
                        name='Test Accuracy',
                        line=dict(color='#10B981', width=2)
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=progress_df['Epoch'],
                        y=progress_df['Loss'],
                        name='Loss',
                        line=dict(color='#EF4444', width=1, dash='dot')
                    ),
                    secondary_y=True
                )
                
                fig.update_layout(
                    title='📊 Model Training Progress',
                    xaxis_title='Epoch',
                    height=500
                )
                
                fig.update_yaxes(title_text="Accuracy", secondary_y=False)
                fig.update_yaxes(title_text="Loss", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Results visualization
                st.subheader("🎯 Model Evaluation")
                
                model.eval()
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1).numpy()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📋 Classification Report")
                    report = classification_report(
                        df['risk_level_encoded'],
                        pred,
                        target_names=label_encoders['risk_level'].classes_,
                        output_dict=True
                    )
                    
                    # Convert to DataFrame for better display
                    report_df = pd.DataFrame(report).transpose()
                    
                    # Add emojis based on f1-score
                    def add_emoji(score):
                        if score >= 0.9:
                            return f"{score:.2f} 🚀"
                        elif score >= 0.7:
                            return f"{score:.2f} ✅"
                        elif score >= 0.5:
                            return f"{score:.2f} 📊"
                        else:
                            return f"{score:.2f} ⚠️"
                    
                    if 'f1-score' in report_df.columns:
                        report_df['f1-score'] = report_df['f1-score'].apply(add_emoji)
                    
                    st.dataframe(report_df, use_container_width=True)
                
                with col2:
                    st.markdown("#### 🧩 Confusion Matrix")
                    cm = confusion_matrix(df['risk_level_encoded'], pred)
                    
                    # Plotly heatmap for confusion matrix
                    fig = px.imshow(
                        cm,
                        x=label_encoders['risk_level'].classes_,
                        y=label_encoders['risk_level'].classes_,
                        text_auto=True,
                        color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual", color="Count")
                    )
                    fig.update_layout(
                        title="🧩 Confusion Matrix",
                        xaxis_title="Predicted Risk Level",
                        yaxis_title="Actual Risk Level",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance visualization
                st.subheader("🔍 Feature Importance Analysis")
                
                # Get weights from the model
                with torch.no_grad():
                    weights = model.lin.weight.cpu().numpy()
                
                # Get top features
                feature_importance = np.abs(weights).mean(axis=0)
                top_indices = np.argsort(feature_importance)[-10:]
                feature_names = (
    list(tfidf_vectorizer.get_feature_names_out()) + 
    ['sentiment', 'impact', 'category', 'source']
)
                top_features = [feature_names[i] for i in top_indices]
                # Visualize
                fig = px.bar(
                    x=feature_importance[top_indices],
                    y=top_features,
                    orientation='h',
                    title="🔍 Top Feature Importance",
                    labels={"x": "Importance", "y": "Feature"},
                    color=feature_importance[top_indices],
                    color_continuous_scale="Viridis"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Welcome screen
        st.markdown("""
        <div class="card" style="text-align: center; padding: 2rem;">
            <h2>👋 Welcome to the Financial Risk Analyzer!</h2>
            <p>Please upload your dataset using the sidebar to get started.</p>
            <div style="font-size: 4rem; margin: 2rem;">
                💰 📊 🚀 📈
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Features description
        st.markdown("""
        <div class="card">
            <h3>✨ Key Features</h3>
            <ul>
                <li>🔍 <strong>Text Analysis</strong> - Process financial news headlines using NLP techniques</li>
                <li>📊 <strong>Interactive Visualizations</strong> - Explore risk patterns with dynamic charts</li>
                <li>🤖 <strong>GNN Model</strong> - Train a Graph Neural Network to predict financial risk</li>
                <li>📈 <strong>Performance Metrics</strong> - Evaluate model accuracy with detailed reports</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        # Footer
        st.markdown("""
<div class="footer">
    <div style="text-align:center;">
        <h3>💰 Financial Risk Analyzer</h3>
        <p style="color:#4e54c8; font-weight:600;">Engineered with ❤️ by Shreyas Kasture for Data Enthusiasts</p>
    </div>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
