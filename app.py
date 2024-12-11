import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
nltk.download('stopwords')

# Load Dataset
def load_data():
    file_path = 'job dataset new.xlsx'
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    return df

# Clean and preprocess text
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

# Extract text from uploaded resume (PDF)
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return clean_text(text)

# Load Pretrained ResNet50 Model
def load_resnet50_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

# Extract Features using ResNet50
def extract_image_features(file, model):
    # Define transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load image and apply transformation
    image = Image.open(file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    # Extract features
    with torch.no_grad():
        features = model(image_tensor)
    return features.flatten().numpy()

# Function to recommend jobs based on resume text and ResNet50
def recommend_jobs_from_resume(resume_text, df, top_n=5):
    # Combine Job Requirements and clean text
    data = df.copy()
    data['Combined_Requirements'] = data['Job_Requirements'].fillna('').apply(clean_text)

    # Append resume text to dataset
    user_data = pd.DataFrame({'Combined_Requirements': [resume_text]})
    combined_data = pd.concat([data[['Combined_Requirements']], user_data], ignore_index=True)

    # TF-IDF Vectorization with additional parameters
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(combined_data['Combined_Requirements'])

    # Compute Cosine Similarity
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Normalize Similarity Scores to fix scale
    cosine_sim = (cosine_sim - cosine_sim.min()) / (cosine_sim.max() - cosine_sim.min())

    # Get top N job recommendations
    data['Similarity_Score'] = cosine_sim[0]
    recommendations = data.sort_values(by='Similarity_Score', ascending=False).head(top_n)
    return recommendations[['Job_title', 'Job_Requirements', 'Similarity_Score']]

# Streamlit App
st.title("Resume-Based Job Recommender System")
st.write("### Upload your resume (PDF) or an optional image for job matching")

# Load ResNet50 Model
resnet50_model = load_resnet50_model()

# Upload Resume
uploaded_file = st.file_uploader("Upload your Resume (PDF only):", type=['pdf'])
image_file = st.file_uploader("Upload an optional Image (JPEG/PNG):", type=['jpg', 'png', 'jpeg'])

# Load the dataset
df = load_data()

# Recommend Jobs
if uploaded_file:
    try:
        # Extract text from PDF
        resume_text = extract_text_from_pdf(uploaded_file)
        st.success("Resume uploaded and processed successfully!")

        # Extract optional image features
        if image_file:
            features = extract_image_features(image_file, resnet50_model)
            st.success("Image uploaded and features extracted successfully!")

        # Get job recommendations
        recommendations = recommend_jobs_from_resume(resume_text, df)
        st.write("### Recommended Jobs for You")
        for i, row in recommendations.iterrows():
            match_percentage = min(row['Similarity_Score'] * 100, 100)  # Ensure percentage does not exceed 100
            st.write(f"**Job Title:** {row['Job_title']}")
            st.write(f"**Requirements:** {row['Job_Requirements']}")
            st.write(f"**Match Percentage:** {match_percentage:.2f}%")
            st.write("---")
    except Exception as e:
        st.error(f"Error processing resume: {e}")
else:
    st.info("Please upload a PDF resume to proceed.")
