# # text_classification.py

# from transformers import pipeline, BertTokenizer, BertForSequenceClassification
# from data_preprocessing import load_and_prepare_data

# # Load the dataset and category mapping from data_preprocessing.py
# _, category_to_id, id_to_category = load_and_prepare_data("data/faq.json")

# # Load the fine-tuned model and tokenizer
# model_path = "./fine_tuned_model"  # Path to the saved fine-tuned model
# tokenizer = BertTokenizer.from_pretrained(model_path)
# model = BertForSequenceClassification.from_pretrained(model_path)

# # Load pipeline for text classification
# text_classification = pipeline("text-classification", model=model, tokenizer=tokenizer)

# # Test the pipeline with an example input
# question = "What is the application process?"
# result = text_classification(question)

# # Extract and convert label
# predicted_label = result[0]['label']  # Expected format: 'LABEL_X'
# predicted_label_index = int(predicted_label.split('_')[1])
# predicted_category = id_to_category[predicted_label_index]

# # Print the actual category and confidence score
# print(f"Predicted Category: {predicted_category}, Confidence Score: {result[0]['score']}")


# from transformers import pipeline, BertTokenizer, BertForSequenceClassification
# from data_preprocessing import load_and_prepare_data

# # Load the dataset and category mapping from data_preprocessing.py
# dataset, category_to_id, id_to_category  = load_and_prepare_data("data/faq.json")

# # Reverse the category_to_id mapping to map label indices back to category names
# id_to_category = {v: k for k, v in category_to_id.items()}

# # Define the model and tokenizer
# model_name = "bert-base-uncased"
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(category_to_id))
# tokenizer = BertTokenizer.from_pretrained(model_name)

# # Load pipeline for text classification
# text_classification = pipeline("text-classification", model=model, tokenizer=tokenizer)

# # Test the pipeline with an example input
# question = "What is the application process?"
# result = text_classification(question)
# # print(result)

# # The result is a list of dictionaries; extract the label
# predicted_label = result[0]['label']  # This will be in the format 'LABEL_X'

# # Extract the numeric index from 'LABEL_X' (e.g., 'LABEL_1' -> 1)
# predicted_label_index = int(predicted_label.split('_')[1])

# # Map the predicted label index back to the actual category name
# predicted_category = id_to_category[predicted_label_index]

# # Print the actual category and confidence score
# print(f"Predicted Category: {predicted_category}, Confidence Score: {result[0]['score']}")


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from data_preprocessing import load_and_prepare_data

# Load model and FAISS index
model, index, texts, labels, category_to_id, id_to_category = load_and_prepare_data("data/faq.json")

def find_best_match(question):
    question_embedding = model.encode([question], normalize_embeddings=True)
    D, I = index.search(np.array(question_embedding), k=1)  # Find top-1 closest match
    return texts[I[0][0]], D[0][0]  # Return best-matched question and its distance

# Example Usage
question = "How do I apply for admission?"
matched_question, similarity_score = find_best_match(question)

print(f"Best Match: {matched_question} \nSimilarity Score: {similarity_score}")
