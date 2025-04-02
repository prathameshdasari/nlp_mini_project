# import json
# import pickle
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from flask import Flask, request, jsonify, send_from_directory

# # Load intent classifier
# with open("models/intent_classifier.pkl", "rb") as f:
#     vectorizer, clf = pickle.load(f)

# # Load FAQ index
# index = faiss.read_index("data/faiss_index.bin")

# # Load embedding model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Load FAQs
# with open("data/faq.json", "r") as f:
#     faq_data = json.load(f)

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return send_from_directory('templates', 'index.html')

# @app.route("/chat", methods=["POST"])
# def chat():
#     user_input = request.json["message"].lower()

#     # Intent classification
#     intent_vector = vectorizer.transform([user_input])
#     intent = clf.predict(intent_vector)[0]

#     # Retrieve best matching FAQ
#     input_embedding = model.encode([user_input])
#     _, idx = index.search(np.array(input_embedding), 1)
#     best_match = faq_data[idx[0][0]]["answer"]

#     return jsonify({"intent": intent, "response": best_match})

# if __name__ == "__main__":
#     app.run(debug=True)


import json
import torch
from flask import Flask, request, jsonify, send_from_directory
from transformers import BertTokenizer, BertForSequenceClassification

# Initialize Flask app
app = Flask(__name__)

# Load fine-tuned model and tokenizer
tokenizer = BertTokenizer.from_pretrained("fine_tuned_model")
model = BertForSequenceClassification.from_pretrained("fine_tuned_model")

# Load FAQs (for response mapping)
with open("data/faq.json", "r") as f:
    faq_data = json.load(f)


@app.route("/")
def home():
    return send_from_directory('templates', 'index.html')


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"].lower()

    # Tokenize input
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()  # Get the predicted class

    # Retrieve corresponding FAQ based on the predicted class
    predicted_faq = faq_data[predicted_class]
    faq_answer = predicted_faq["faqs"][0]["answer"]  # For simplicity, return the first FAQ answer

    return jsonify({"response": faq_answer})

if __name__ == "__main__":
    app.run(debug=True)


