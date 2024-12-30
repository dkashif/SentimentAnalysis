import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import string

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("vader_lexicon")
nltk.download("wordnet")
nltk.download("punkt_tab")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

df = pd.read_csv("C:/Users/daniyalkashif2/Downloads/IMDB Dataset/IMDB-Dataset.csv")


# Display the first few rows of the dataset
print("Original DataFrame:")
print(df.head())


# Define the preprocessing function
def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )  # Remove punctuation
    words = word_tokenize(text)  # Tokenize words
    # Remove stopwords and lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)


# Apply preprocessing to the 'user_review' column
df["cleaned_review"] = df["review"].astype(str).apply(preprocess)

# Display the cleaned text
print("\nDataFrame with Cleaned Reviews:")
print(df[["review", "cleaned_review"]].head())


# Feature Extraction: Bag of Words

# Initialize the TfidfVectorizer with a limited vocabulary size
vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the cleaned text data
X = vectorizer.fit_transform(df["cleaned_review"])

# Display the feature matrix shape
print("\nFeature Matrix Shape:")
print(X.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, df["sentiment"], test_size=0.2, random_state=42
)

# Initialize the classifier
classifier = LogisticRegression()

# Train the classifier
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
