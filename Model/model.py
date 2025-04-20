# -*- coding: utf-8 -*-
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

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Set page config
st.set_page_config(page_title="Financial Risk Analyzer", layout="wide")

# Helper functions
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return ' '.join(tokens)

def main():
    st.title("Graph Neural Networks for Financial Risk Analysis")
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    SAMPLE_DATA = "Dataset/financial_risk_dataset_enhanced.csv"
    csv = SAMPLE_DATA.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download Sample Dataset",
        data=csv,
        file_name="financial_risk_dataset.csv",
        mime="text/csv"
    )
    uploaded_file = st.sidebar.file_uploader("Please Upload Your Financial-Risk Dataset", type=["csv"])
    epochs = st.sidebar.slider("Training Epochs", 10, 300, 200)
    hidden_channels = st.sidebar.slider("Hidden Channels", 16, 128, 64)

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        # Data Overview Section
        st.header("Data Overview")
        st.write("Dataset shape:", df.shape)
        
        if st.checkbox("Show raw data"):
            st.subheader("Raw Data")
            st.write(df.head())

        # Preprocessing
        st.header("Text Preprocessing")
        df['processed_text'] = df['headline'].apply(preprocess_text)
        
        if st.checkbox("Show processed text examples"):
            for i in range(3):
                st.write(f"Original: {df['headline'].iloc[i]}")
                st.write(f"Processed: {df['processed_text'].iloc[i]}")
                st.write("---")

        # TF-IDF Processing
        tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])
        
        # Visualization
        st.header("Data Visualizations")
        viz_option = st.selectbox("Choose visualization", 
                                ["TF-IDF Terms", "Category Distribution", 
                                 "Risk Level Distribution", "Sentiment Analysis"])

        if viz_option == "TF-IDF Terms":
            tfidf_sum = pd.DataFrame(tfidf_matrix.sum(axis=0).T,
                                    index=tfidf_vectorizer.get_feature_names_out(),
                                    columns=['sum']).sort_values('sum', ascending=False)
            plt.figure(figsize=(12,6))
            tfidf_sum.head(20).plot(kind='bar')
            st.pyplot(plt)
        
        elif viz_option == "Category Distribution":
            fig, ax = plt.subplots(figsize=(10,6))
            df['category'].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)
            
        elif viz_option == "Risk Level Distribution":
            fig, ax = plt.subplots(figsize=(10,6))
            df['risk_level'].value_counts().plot(kind='bar', ax=ax)
            ax.set_title('Risk Level Distribution')
            ax.set_xlabel('Risk Level')
            ax.set_ylabel('Count')
            st.pyplot(fig)
            
        elif viz_option == "Sentiment Analysis":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
            
            # Sentiment Score by Category
            sns.boxplot(x='category', y='sentiment_score', data=df, ax=ax1)
            ax1.set_title('Sentiment Score by Category')
            ax1.tick_params(axis='x', rotation=45)
            
            # Impact Score by Category
            sns.boxplot(x='category', y='impact_score', data=df, ax=ax2)
            ax2.set_title('Impact Score by Category')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)

        # Model Training Section
        st.header("GNN Model Training")
        
        if st.button("Train Model"):
            st.write("## Training Progress")
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
                    self.conv2 = GCNConv(hidden_channels, hidden_channels)
                    self.lin = torch.nn.Linear(hidden_channels, num_classes)

                def forward(self, x, edge_index):
                    x = self.conv1(x, edge_index).relu()
                    x = self.conv2(x, edge_index).relu()
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
            
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = criterion(out[train_mask], data.y[train_mask])
                loss.backward()
                optimizer.step()

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
                status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} | Train Acc: {train_acc:.2f} | Test Acc: {test_acc:.2f}")

            st.success("Training completed!")
            
            # Show accuracy graph
            st.subheader("Training Progress")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(train_accuracies, label='Train Accuracy')
            ax.plot(test_accuracies, label='Test Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Training and Validation Accuracy')
            ax.legend()
            st.pyplot(fig)
            
            # Show results
            model.eval()
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1).numpy()
            
            st.subheader("Classification Report")
            st.text(classification_report(df['risk_level_encoded'], pred,
                                        target_names=label_encoders['risk_level'].classes_))

            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(10,8))
            cm = confusion_matrix(df['risk_level_encoded'], pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                        xticklabels=label_encoders['risk_level'].classes_,
                        yticklabels=label_encoders['risk_level'].classes_)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
