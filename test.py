import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("vader_lexicon")
nltk.download("wordnet")
nltk.download("punkt_tab")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

df = pd.read_csv("C:/Users/daniyalkashif2/Downloads/train_gr/train.csv")


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
df["cleaned_review"] = df["user_review"].astype(str).apply(preprocess)

# Display the cleaned text
print("\nDataFrame with Cleaned Reviews:")
print(df[["user_review", "cleaned_review"]].head())
