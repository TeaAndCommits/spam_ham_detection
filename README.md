# spam_ham_detection
Detect spam emails with Python using NLP preprocessing and a Random Forest classifier.

## Table of Contents
- [About](#about)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Explanation](#explanation)
- [Deployment](#deployment)
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
- Streamlit

## Dataset
The dataset used is `spam_ham_dataset.csv`.  
It contains:
- `text`: the email content
- `label`: spam or ham
- `label_num`: numeric representation of the label (0 = ham, 1 = spam)


## Explanation

### 1. Importing Libraries
We start by importing Python libraries :
- pandas — for data handling

- numpy — for numerical operations

- nltk — for natural language processing (stopwords, stemming)

- scikit-learn — for machine learning (vectorizer, classifier)

- streamlit — for the web app interface

---

### 2. Loading the Dataset
```python
df = pd.read_csv('spam_ham_dataset.csv')
df['text'] = df['text'].apply(lambda x: x.replace('\f\n', ' '))
```
Reads the dataset from a CSV file.

Cleans unwanted line breaks from the email text.

### 3. Preprocessing the Text

We apply several preprocessing steps:

Lowercasing: Makes all words the same case (Spam → spam).

Punctuation removal: Removes symbols like !, ,, ?.

Stopword removal: Removes common words like "the", "is", "and".

Stemming: Converts words to their root form (running → run).
```python
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))
corpus = []

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    return ' '.join(text)

corpus = [preprocess(text) for text in df['text']]
```

### 4. Converting Text to Numbers
Machine learning models can’t understand raw text directly — we need to convert it into numerical features.
We use Bag of Words with CountVectorizer:
```python
vectorizer = CountVectorizer(max_features=5000) 
X = vectorizer.fit_transform(corpus)  # keep it sparse, no .toarray()
y = df.label_num
```
x: Matrix of word counts for each email.

y: Labels (0 = ham, 1 = spam).

### 5. Train model
We use a Random Forest Classifier:
```python
clf = RandomForestClassifier(n_jobs=-1)
clf.fit(X, y)
```
n_jobs=-1 → uses all CPU cores for faster training.
### 6. Streamlit UI

```python
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
```


## Deployment
https://spamhamdetection-bp3c5pczcspga8tmpagj4k.streamlit.app/
Click on the link to check the streamlit app 

## Future Improvements
- Experiment with other ML models (e.g., Naive Bayes, SVM)





