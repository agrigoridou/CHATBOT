import json, pickle, random
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from colorama import Fore, Style, init

init()

model = load_model("model/chat_model")
tokenizer = pickle.load(open("model/tokenizer.pickle", "rb"))
encoder = pickle.load(open("model/label_encoder.pickle", "rb"))

with open("data/intents.json") as f:
    data = json.load(f)

print(Fore.YELLOW + "Chatbot started (type 'quit')" + Style.RESET_ALL)

while True:
    msg = input(Fore.BLUE + "You: " + Style.RESET_ALL)
    if msg.lower() == "quit":
        break

    seq = tokenizer.texts_to_sequences([msg])
    padded = pad_sequences(seq, maxlen=20, truncating="post")
    prediction = model.predict(padded)
    tag = encoder.inverse_transform([np.argmax(prediction)])[0]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            print(Fore.GREEN + "Bot: " + random.choice(intent["responses"]))
