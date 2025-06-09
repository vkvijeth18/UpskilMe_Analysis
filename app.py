from flask import Flask, request, jsonify
from flask_cors import CORS
from analysis import analyze_video
import os
import nltk
import textblob.download_corpora

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Ensure TextBlob corpora are available
try:
    from textblob import TextBlob
    _ = TextBlob("test").sentiment
except Exception:
    print("Downloading TextBlob corpora...")
    textblob.download_corpora.download_all()
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/", methods=["GET"])
def index():
     return "pong", 200

@app.route("/ping",methods=["GET"])
def ping():
    return "Server is Up & Running."

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        video_url = data.get("videoUrl")
        interview_id = data.get("interviewId")

        if not video_url or not interview_id:
            return jsonify({"error": "Missing videoUrl or interviewId"}), 400

        result = analyze_video(video_url)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
