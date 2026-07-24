from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model/fake_job_detector.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    text = request.form["job"]

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    probability = model.predict_proba(text_vector)[0]

    confidence = max(probability) * 100

    # Suspicious keywords
    keywords = [
        "registration fee",
        "telegram",
        "whatsapp",
        "pay",
        "deposit",
        "urgent",
        "guaranteed",
        "no interview"
    ]

    found = []

    for word in keywords:
        if word.lower() in text.lower():
            found.append(word)

    if prediction == 1:
        result = f"⚠️ Fake Job ({confidence:.2f}% confidence)"
    else:
        result = f"✅ Real Job ({confidence:.2f}% confidence)"

    return render_template(
        "index.html",
        prediction=result,
        reasons=found
    )
import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)
