import json
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load data
with open("data/intents.json") as file:
    data = json.load(file)

sentences, labels = [], []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])

# Encode labels
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

# Tokenization
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=20, truncating="post")

# Model
model = Sequential([
    Embedding(1000, 16, input_length=20),
    GlobalAveragePooling1D(),
    Dense(16, activation="relu"),
    Dense(len(set(labels)), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(padded, np.array(encoded_labels), epochs=300)

# Save
model.save("model/chat_model")
pickle.dump(tokenizer, open("model/tokenizer.pickle", "wb"))
pickle.dump(encoder, open("model/label_encoder.pickle", "wb"))

print("✅ Training completed")
