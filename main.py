from dataset import load_data
from preprocess import clean_text

df = load_data("data/IMDB Dataset.csv")

print("Original review:")
print(df["review"][0])

print("\nCleaned review:")
print(clean_text(df["review"][0]))