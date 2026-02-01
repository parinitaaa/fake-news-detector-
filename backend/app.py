from flask import Flask, request, jsonify
import joblib
import re
import string

svm_model = joblib.load("fake_news_svm.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text.strip()

app = Flask(__name__)

@app.route("/")
def home():
    return "Fake News Detection API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    pred = svm_model.predict(vec)[0]

    result = "FAKE" if pred == 0 else "REAL"

    return jsonify({
        "prediction": result
    })

if __name__ == "__main__":
    app.run(debug=True)
