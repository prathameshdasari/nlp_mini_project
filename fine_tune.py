# import torch
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# from data_preprocessing import load_and_prepare_data
# from datasets import DatasetDict

# # Load dataset and category mapping
# dataset, category_to_id, id_to_category = load_and_prepare_data("data/faq.json")

# # Define model and tokenizer
# model_name = "bert-base-uncased"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(category_to_id))

# # Tokenization function
# def tokenize_function(example):
#     return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

# # Apply tokenization
# tokenized_dataset = dataset.map(tokenize_function, batched=True)

# # Rename `label` to `labels` (Hugging Face Trainer expects this)
# tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

# # Convert dataset to PyTorch format
# tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# # Split into train and test (80% train, 20% test)
# split = tokenized_dataset.train_test_split(test_size=0.2)
# train_dataset = split["train"]
# test_dataset = split["test"]

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     logging_dir="./logs",
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     save_total_limit=2
# )

# # Define Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset
# )

# # Train the model
# trainer.train()

# # Save the fine-tuned model
# model.save_pretrained("./fine_tuned_model")
# tokenizer.save_pretrained("./fine_tuned_model")


import torch
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from data_preprocessing import load_and_prepare_data
from torch.utils.data import Dataset

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load dataset and embeddings from data_preprocessing.py
model, index, texts, labels, category_to_id, id_to_category = load_and_prepare_data("data/faq.json")

# Set the number of labels based on the unique categories in the data
num_labels = len(category_to_id)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

# Split data into train and eval sets (80% train, 20% eval)
train_texts, eval_texts, train_labels, eval_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Define a custom dataset class
class FAQDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Tokenize the text
        encoding = self.tokenizer(
            self.texts[idx], 
            truncation=True,
            padding="max_length", 
            max_length=self.max_length,  # BERT max length is 512
            return_tensors="pt",  # Returns PyTorch tensors
        )

        # Return the tokenized input and label
        return {
            "input_ids": encoding["input_ids"].squeeze(),  # Remove the batch dimension
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx]),  # Ensure labels are tensor
        }

# Create datasets
train_dataset = FAQDataset(train_texts, train_labels, tokenizer)
eval_dataset = FAQDataset(eval_texts, eval_labels, tokenizer)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Evaluate after every epoch
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",  # Save after every epoch
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Start fine-tuning the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model2")





