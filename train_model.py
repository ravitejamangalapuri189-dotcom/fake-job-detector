import pandas as pd
df = pd.read_csv("dataset/fake_job_postings.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.columns.tolist())
df = df.fillna("")
df["text"] = (
    df["title"] + " " +
    df["company_profile"] + " " +
    df["description"] + " " +
    df["requirements"]
)
df = df[["text", "fraudulent"]]
df["text"] = df["text"].str.lower()
import re
def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text
df["text"] = df["text"].apply(clean_text)
df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)
print(df.head())
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

X = vectorizer.fit_transform(df["text"])

y = df["fraudulent"]

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("Model trained successfully!")
from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
import joblib

# Save the trained model
joblib.dump(model, "model/fake_job_detector.pkl")

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model and vectorizer saved successfully!")
from sklearn.metrics import classification_report, confusion_matrix

predictions = model.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
