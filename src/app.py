import os
from flask import Flask, request, jsonify, render_template
import json, pickle, numpy as np, re, random
from tensorflow.keras.models import load_model
import spacy

# -----------------------
# Initialize Flask and models
# -----------------------


template_path = os.path.join(os.path.dirname(__file__), "../templates")
app = Flask(__name__, template_folder=template_path)

@app.route("/")
def home():
    return render_template("index.html")

# Load trained ML model
model = load_model("model/chat_model")
tokenizer = pickle.load(open("model/tokenizer.pickle", "rb"))
encoder = pickle.load(open("model/label_encoder.pickle", "rb"))

# Load intents
with open("data/intents.json") as f:
    data = json.load(f)

# Load spaCy English model for semantic similarity
nlp = spacy.load("en_core_web_md")

# -----------------------
# Helper functions
# -----------------------
def clean_text(text):
    return re.sub(r"[^\w\s]", "", text.lower())

def get_best_intent(msg):
    """Fallback using semantic similarity (spaCy)"""
    msg_doc = nlp(msg)
    best_score = 0
    best_tag = None
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            pattern_doc = nlp(pattern)
            similarity = msg_doc.similarity(pattern_doc)
            if similarity > best_score:
                best_score = similarity
                best_tag = intent["tag"]
    return best_tag, best_score

# -----------------------
# Flask route
# -----------------------
@app.route("/chat", methods=["POST"])
def chat():
    msg = request.json.get("message")
    if not msg:
        return jsonify({"response": "Please send a message."})

    msg_clean = clean_text(msg)
    seq = tokenizer.texts_to_sequences([msg_clean])
    padded = np.array(seq)
    
    # Predict with ML model
    prediction = model.predict(padded)
    confidence = np.max(prediction)
    tag_index = np.argmax(prediction)

    # High confidence -> use model prediction
    if confidence >= 0.6:
        tag = encoder.inverse_transform([tag_index])[0]
    else:
        # Low confidence -> fallback to semantic similarity
        tag, score = get_best_intent(msg_clean)
        if score < 0.6:
            return jsonify({"response": "I'm not sure I understood that 🤔 Could you rephrase?"})

    # Return random response from intent
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return jsonify({"response": random.choice(intent["responses"])})

# -----------------------
# Run app
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)
