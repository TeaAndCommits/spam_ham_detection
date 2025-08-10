import streamlit as st
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

nltk.download('stopwords')

# Load and preprocess your dataset once
df = pd.read_csv('spam_ham_dataset.csv')
df['text'] = df['text'].apply(lambda x: x.replace('\f\n', ' '))

# Preprocess function (same as your loop)
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    return ' '.join(text)

corpus = [preprocess(text) for text in df['text']]

# Vectorize
vectorizer = CountVectorizer(max_features=5000) 
X = vectorizer.fit_transform(corpus)  # keep it sparse, no .toarray()

y = df.label_num

# Train model
clf = RandomForestClassifier(n_jobs=-1)
clf.fit(X, y)

# Streamlit UI
st.title("Spam Detector")

user_input = st.text_area("Enter your email/message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        processed_input = preprocess(user_input)
        vectorized_input = vectorizer.transform([processed_input]).toarray()
        prediction = clf.predict(vectorized_input)[0]
        label = "Spam" if prediction == 1 else "Not Spam"
        st.success(f"The message is classified as: {label}")
