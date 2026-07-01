# Movie Review Sentiment Analysis

A beginner-friendly machine learning project that predicts whether a movie review is **positive** or **negative**.

The project uses the IMDb movie review dataset, cleans the review text, trains a Logistic Regression model with TF-IDF features, and provides a simple Streamlit web app for predictions.

## Features

- Load and validate the IMDb dataset
- Clean review text by removing HTML tags, URLs, punctuation, numbers, extra spaces, and stopwords
- Preserve important negation words like `not`, `no`, and `never`
- Train a sentiment classification model
- Save the trained model and vectorizer
- Predict sentiment for new reviews
- Use a Streamlit app for an easy web interface

## Project Structure

```text
sentiment-analysis/
|-- app.py
|-- dataset.py
|-- preprocess.py
|-- train.py
|-- predict.py
|-- main.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- IMDB Dataset.csv
|-- saved_model/
|   |-- model.pkl
|   |-- vectorizer.pkl
```

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv venv
.\venv\Scripts\activate
```

2. Install the required packages.

```powershell
pip install -r requirements.txt
```

3. Add the IMDb dataset file inside the `data` folder.

```text
data/IMDB Dataset.csv
```

The CSV file should contain these columns:

- `review`
- `sentiment`

## Train the Model

Run:

```powershell
python train.py
```

This will:

- Load the dataset
- Clean all reviews
- Train the model
- Print accuracy and a classification report
- Save the trained files inside `saved_model/`

## Test Prediction

Run:

```powershell
python predict.py
```

This runs sample predictions using the saved model.

## Run the Streamlit App

Run:

```powershell
streamlit run app.py
```

Then open the local Streamlit URL in your browser and enter a movie review.

## Model Details

- Text features: `TfidfVectorizer`
- Model: `LogisticRegression`
- Dataset: IMDb movie reviews
- Output labels: `positive` or `negative`

## Notes

The `data/` and `saved_model/` folders are ignored by Git because datasets and model files can be large. To recreate the model, place the dataset in `data/` and run `python train.py`.
