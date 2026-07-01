import pickle
import re

from preprocess import clean_text


MODEL_PATH = "saved_model/model.pkl"
VECTORIZER_PATH = "saved_model/vectorizer.pkl"
POSITIVE_PHRASES = (
    "not bad",
    "not terrible",
    "not worst",
    "not the worst",
    "not boring",
    "not disappointing",
)


NEGATIVE_PHRASES = (
    "dont like",
    "didnt like",
    "do not like",
    "did not like",
    "bad movie",
    "terrible movie",
    "worst movie",
)
NEGATION_PATTERNS = (
    r"\bnot\s+(?:a\s+|an\s+|that\s+|so\s+|very\s+)?(?:good|great|nice|amazing|excellent|worth|recommended)\b",
    r"\bnever\s+(?:a\s+|an\s+|that\s+|so\s+|very\s+)?(?:good|great|nice|amazing|excellent|worth|recommended)\b",
    r"\bno\s+(?:good|fun|value|sense)\b",
)


def load_pickle_file(file_path):
  
    with open(file_path, "rb") as file:
        return pickle.load(file)


def has_positive_phrase(text):
    """Check whether text contains a clear positive phrase."""
    normalized_text = text.lower().replace("'", "")

    return any(phrase in normalized_text for phrase in POSITIVE_PHRASES)


def has_negative_phrase(text):
    """Check whether text contains a clear negative phrase."""
    normalized_text = text.lower().replace("'", "")

    if any(phrase in normalized_text for phrase in NEGATIVE_PHRASES):
        return True

    return any(re.search(pattern, normalized_text) for pattern in NEGATION_PATTERNS)


def has_low_rating(text):
  
    normalized_text = text.lower()
    rating_patterns = (
        r"(?:rating it|rating|rated)\s*(\d+)",
        r"\b(\d+)\s*/\s*10\b",
    )

    for pattern in rating_patterns:
        rating_match = re.search(pattern, normalized_text)
        if rating_match and int(rating_match.group(1)) <= 4:
            return True

    return False


model = load_pickle_file(MODEL_PATH)
vectorizer = load_pickle_file(VECTORIZER_PATH)


def predict_sentiment(text):
   
    if not isinstance(text, str) or not text.strip():
        return "negative"

    if has_positive_phrase(text):
        return "positive"

    if has_negative_phrase(text) or has_low_rating(text):
        return "negative"

    cleaned_text = clean_text(text)

    if not cleaned_text:
        return "negative"

    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)[0]

    if prediction == 1:
        return "positive"

    return "negative"


if __name__ == "__main__":
    sample_reviews = [
        "This movie was excellent and I really enjoyed it!",
        "not bad",
        "not a good movie",
        "not a good movie rating it 3",
    ]

    for sample_review in sample_reviews:
        sentiment = predict_sentiment(sample_review)
        print(f"Review: {sample_review}")
        print(f"Predicted sentiment: {sentiment}")
        print()
