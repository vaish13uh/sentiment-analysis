"""Train and save the sentiment analysis model."""

import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from dataset import load_data
from preprocess import clean_text


DATA_PATH = "data/IMDB Dataset.csv"
MODEL_DIR = "saved_model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")


def prepare_data(dataframe):
    """Clean reviews and convert sentiment labels into numeric values.

    Args:
        dataframe (pandas.DataFrame): Dataset with review and sentiment columns.

    Returns:
        tuple: Cleaned review text and numeric sentiment labels.
    """
    reviews = dataframe["review"].apply(clean_text)
    labels = dataframe["sentiment"].map({"positive": 1, "negative": 0})

    return reviews, labels


def save_pickle_file(data, file_path):
    """Save data to a pickle file.

    Args:
        data: Python object to save.
        file_path (str): Destination path for the pickle file.
    """
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def train_model():
    """Train the sentiment analysis model and save model artifacts."""
    dataframe = load_data(DATA_PATH)
    reviews, labels = prepare_data(dataframe)

    x_train, x_test, y_train, y_test = train_test_split(
        reviews,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    x_train_vectors = vectorizer.fit_transform(x_train)
    x_test_vectors = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_vectors, y_train)

    predictions = model.predict(x_test_vectors)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("Classification report:")
    print(classification_report(y_test, predictions))

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_pickle_file(model, MODEL_PATH)
    save_pickle_file(vectorizer, VECTORIZER_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")


if __name__ == "__main__":
    train_model()
