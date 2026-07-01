

import re
import string

import nltk
from nltk.corpus import stopwords


STOPWORDS = None
NEGATION_WORDS = {
    "no",
    "nor",
    "not",
    "never",
    "none",
    "n't",
    "dont",
    "don't",
    "didnt",
    "didn't",
    "isnt",
    "isn't",
    "wasnt",
    "wasn't",
    "wont",
    "won't",
    "cant",
    "can't",
    "couldnt",
    "couldn't",
}


def _load_stopwords():
   
    try:
        english_stopwords = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        english_stopwords = set(stopwords.words("english"))

    return english_stopwords - NEGATION_WORDS


def _get_stopwords():
    
    global STOPWORDS

    if STOPWORDS is None:
        STOPWORDS = _load_stopwords()

    return STOPWORDS


def clean_text(text):
    
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [word for word in text.split() if word not in _get_stopwords()]

    return " ".join(words)
