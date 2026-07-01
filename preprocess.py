"""Text preprocessing utilities for sentiment analysis."""

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
    """Load English stopwords, downloading them only when needed."""
    try:
        english_stopwords = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        english_stopwords = set(stopwords.words("english"))

    return english_stopwords - NEGATION_WORDS


def _get_stopwords():
    """Return cached English stopwords."""
    global STOPWORDS

    if STOPWORDS is None:
        STOPWORDS = _load_stopwords()

    return STOPWORDS


def clean_text(text):
    """Clean raw review text for sentiment analysis.

    Args:
        text (str): Raw text to clean.

    Returns:
        str: Cleaned text, or an empty string for non-string inputs.
    """
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
