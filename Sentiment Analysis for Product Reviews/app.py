import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import streamlit as st
import nltk

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Step 1: Load Dataset
def load_data():
    try:
        df = pd.read_csv("Reviews.csv")  # Change to your uploaded dataset file name
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'Reviews.csv' is in the same directory.")
        st.stop()

    # Create sentiment column based on Score
    df = df[['Score', 'Text']].dropna()
    df['sentiment'] = df['Score'].apply(lambda x: 'positive' if x > 3 else ('neutral' if x == 3 else 'negative'))
    df.rename(columns={'Text': 'review_text'}, inplace=True)
    return df

# Step 2: Clean the Text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Step 3: Tokenization
def tokenize_text(df):
    df['tokens'] = df['cleaned_text'].apply(word_tokenize)
    return df

# Step 4: EDA and Visualization
def eda_visualization(df):
    st.subheader("Exploratory Data Analysis")
    sentiment_distribution = df['sentiment'].value_counts()
    st.bar_chart(sentiment_distribution)

    st.subheader("Word Cloud for Positive Reviews")
    positive_reviews = ' '.join(df[df['sentiment'] == 'positive']['cleaned_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
    st.image(wordcloud.to_array())

# Step 5: Train Model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Model Performance")
    st.text(classification_report(y_test, y_pred))
    return model

# Main App
stop_words = set(stopwords.words('english'))

def main():
    st.title(" Sentiment Analysis for Product Reviews")

    df = load_data()
    st.write(" Dataset Loaded Successfully!")
    st.dataframe(df.head())

    # Preprocessing
    df['cleaned_text'] = df['review_text'].apply(clean_text)
    df = tokenize_text(df)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['sentiment']

    # EDA Visualization
    eda_visualization(df)

    # Model Training
    model = train_model(X, y)

    # Prediction Section
    st.subheader("Try It Yourself!")
    user_input = st.text_area("Enter a product review:")
    if st.button("Analyze Sentiment"):
        user_vector = vectorizer.transform([clean_text(user_input)])
        prediction = model.predict(user_vector)[0]
        st.success(f"Predicted Sentiment: **{prediction.upper()}**")

if __name__ == "__main__":
    main()

