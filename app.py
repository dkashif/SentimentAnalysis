from flask import Flask, request, jsonify
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

app = Flask(__name__)


# Load the trained model and vectorizer
classifier = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
