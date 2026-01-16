# Sentiment-Analysis-for-Product-Reviews
 Project Overview

This project focuses on analyzing customer product reviews and classifying them into positive, neutral, or negative sentiments using Natural Language Processing (NLP) and Machine Learning techniques. The system provides insights into customer opinions and overall product perception. 

Sentiment Analysis for Product …

Objective

Develop a machine learning system to analyze customer reviews
Classify reviews into positive, neutral, or negative sentiments
Visualize sentiment distribution and important keywords
Enable real-time sentiment prediction through a user interface

Technologies Used

Programming Language: Python
Framework: Streamlit

Libraries:

Pandas
NLTK
Scikit-learn
Matplotlib
WordCloud

Dataset

Product review dataset (Reviews.csv)

Key columns used:

Score
Text

Sentiments are derived from review scores:

Score > 3 → Positive
Score = 3 → Neutral
Score < 3 → Negative

System Workflow

Load and preprocess dataset
Clean review text (remove HTML, URLs, special characters)
Tokenize text and remove stopwords
Convert text to numerical features using TF-IDF Vectorization
Train a Logistic Regression model
Evaluate model performance
Predict sentiment for user-entered reviews

 Machine Learning Model

Algorithm Used: Logistic Regression
Evaluation Metrics: Precision, Recall, F1-score, Accuracy

 Features

Exploratory Data Analysis (EDA)
Sentiment distribution bar chart
Word cloud for positive reviews
Real-time sentiment prediction
Interactive Streamlit web interface

 How to Run the Project

Install required libraries

pip install pandas nltk scikit-learn matplotlib wordcloud streamlit

Place Reviews.csv in the project directory

Run the application

streamlit run app.py


Open the browser at http://localhost:8501

📈 Conclusion

The project demonstrates how NLP and machine learning can effectively analyze customer opinions. It helps businesses understand customer sentiment, improve products, and make data-driven decisions.
