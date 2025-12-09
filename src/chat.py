import json, pickle, random, re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from colorama import Fore, Style, init
import spacy

init()

# Load model and objects
model = load_model("model/chat_model")
tokenizer = pickle.load(open("model/tokenizer.pickle", "rb"))
encoder = pickle.load(open("model/label_encoder.pickle", "rb"))

with open("data/intents.json") as f:
    data = json.load(f)

# Load spaCy model
nlp = spacy.load("en_core_web_md")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def get_best_intent(msg):
    """Semantic similarity fallback using spaCy"""
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

print(Fore.YELLOW + "Chatbot started (type 'quit')" + Style.RESET_ALL)

while True:
    msg = input(Fore.BLUE + "You: " + Style.RESET_ALL)
    if msg.lower() == "quit":
        break

    msg_clean = clean_text(msg)
    seq = tokenizer.texts_to_sequences([msg_clean])
    padded = pad_sequences(seq, maxlen=20, truncating="post")
    prediction = model.predict(padded)

    confidence = np.max(prediction)
    tag_index = np.argmax(prediction)

    # If model confidence is high enough, use model prediction
    if confidence >= 0.6:
        tag = encoder.inverse_transform([tag_index])[0]
    else:
        # fallback: semantic similarity
        tag, score = get_best_intent(msg_clean)
        if score < 0.6:
            print(Fore.GREEN + "Bot: " + Style.RESET_ALL + "I'm not sure I understood that 🤔 Could you rephrase?")
            continue

    for intent in data["intents"]:
        if intent["tag"] == tag:
            print(Fore.GREEN + "Bot: " + Style.RESET_ALL + random.choice(intent["responses"]))
