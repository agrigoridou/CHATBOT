import json
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re

# -----------------------
# Load and preprocess data
# -----------------------
with open("data/intents.json") as file:
    data = json.load(file)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

sentences, labels = [], []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(clean_text(pattern))
        labels.append(intent["tag"])

# Encode labels
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

# Tokenization
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=20, truncating="post")

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    padded, encoded_labels, test_size=0.2, random_state=42
)

# -----------------------
# Model
# -----------------------
model = Sequential([
    Embedding(2000, 32, input_length=20),
    GlobalAveragePooling1D(),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(len(set(labels)), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=300)

# Evaluate
y_pred = model.predict(X_val).argmax(axis=1)
print("\nClassification Report:\n")
print(classification_report(y_val, y_pred))

# -----------------------
# Save
# -----------------------
model.save("model/chat_model")
pickle.dump(tokenizer, open("model/tokenizer.pickle", "wb"))
pickle.dump(encoder, open("model/label_encoder.pickle", "wb"))

print("✅ Training completed")
