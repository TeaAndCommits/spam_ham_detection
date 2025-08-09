# spam_ham_detection
Detect spam emails with Python using NLP preprocessing and a Random Forest classifier.

## Table of Contents
- [About](#about)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Future Improvements](#future-improvements)


## About
This project classifies emails as either "Spam" or "Ham" (not spam) using machine learning.  
It demonstrates:
- Text preprocessing (lowercasing, punctuation removal, stopword removal, stemming)
- Feature extraction with Bag-of-Words
- Classification using Random Forest


## Features
- Cleans and preprocesses email text
- Converts text into numerical features
- Trains a Random Forest model
- Achieves high accuracy on test data
- Easily test with your own email samples


## Tech Stack
- Python 
- Pandas, NumPy
- NLTK
- Scikit-learn

## Dataset
The dataset used is `spam_ham_dataset.csv`.  
It contains:
- `text`: the email content
- `label`: spam or ham
- `label_num`: numeric representation of the label (0 = ham, 1 = spam)

## Future Improvements
- Try TF-IDF Vectorizer instead of CountVectorizer
- Experiment with other ML models (e.g., Naive Bayes, SVM)
- Deploy as a web app





