import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_and_prepare_data(json_path):
    # Load the FAQ data
    with open(json_path, "r") as f:
        faq_data = json.load(f)

    # Initialize the SentenceTransformer model
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    texts = []
    labels = []
    category_to_id = {category['category']: idx for idx, category in enumerate(faq_data)}
    id_to_category = {v: k for k, v in category_to_id.items()}

    # Extract text variations and create labels for each category
    for category in faq_data:
        for faq in category['faqs']:
            for variation in faq['variations']:
                texts.append(variation)
                labels.append(category_to_id[category['category']])

    # Generate sentence embeddings using the SentenceTransformer model
    embeddings = model.encode(texts, normalize_embeddings=True)

    # Store the embeddings in a FAISS index
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings))

    return model, index, texts, labels, category_to_id, id_to_category
