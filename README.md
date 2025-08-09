# spam_ham_detection
Detect spam emails with Python using NLP preprocessing and a Random Forest classifier.

## Table of Contents
- [About](#about)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Explanation](#explanation)
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


## Explanation

### 1. Importing Libraries
We start by importing Python libraries for:
- **Data handling**: `pandas`, `numpy`
- **Text processing**: `nltk`, `string`
- **Machine learning**: `scikit-learn`

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

for i in range(len(df)):
    text = df['text'].iloc[i].lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = ' '.join(text)
    corpus.append(text)
```

### 4. Converting Text to Numbers
Machine learning models can’t understand raw text directly — we need to convert it into numerical features.
We use Bag of Words with CountVectorizer:
```python
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(corpus).toarray()
y = df.label_num
x: Matrix of word counts for each email.

y: Labels (0 = ham, 1 = spam).
```

### 5. Splitting into Train/Test Sets
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```
80% of the data is used for training.

20% is used for testing model accuracy.

### 6. Training the Model
We use a Random Forest Classifier:
```python
clf = RandomForestClassifier(n_jobs=-1)
clf.fit(x_train, y_train)
```
Builds multiple decision trees and uses majority voting for prediction.

n_jobs=-1 → uses all CPU cores for faster training.

### 7. Evaluating Accuracy
```python
score = clf.score(x_test, y_test)
print("Accuracy:", score)
```
Checks how well the model performs on unseen test data.

### 8. Predicting New Emails
We can pass a new email (after preprocessing and vectorizing it) to:
```python
clf.predict([vectorized_email])
```
Returns 1 for Spam and 0 for Ham.

## Future Improvements
- Try TF-IDF Vectorizer instead of CountVectorizer
- Experiment with other ML models (e.g., Naive Bayes, SVM)
- Deploy as a web app





