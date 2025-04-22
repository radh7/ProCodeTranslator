import json
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
import evaluate
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Paths
model_path = "./models/t5-finetuned"

# Load config
with open("config.json") as f:
    config = json.load(f)

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

# Load the validation dataset
validation_dataset = Dataset.from_json("./data/validation.json")

# Language keys mapping
LANG_KEY_MAP = {
    "java": "java",
    "c#": "cs",
    "csharp": "cs"
}
print(f"Using device: {device}")
# Load BLEU and ROUGE metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

# Function to compute Exact Match (EM)
def exact_match(predictions, references):
    exact_matches = sum([pred == ref for pred, ref in zip(predictions, references)])
    return exact_matches / len(references)

# Preprocess the data (tokenize inputs and targets)
def preprocess_function(examples):
    inputs = examples["java"]
    targets = examples["cs"]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

# Tokenize the validation dataset
tokenized_dataset = validation_dataset.map(preprocess_function, batched=True)

# Create DataLoader for batching
batch_size = config['batch_size']
data_loader = DataLoader(tokenized_dataset, batch_size=batch_size)

# Generate predictions
def generate_predictions(model, data_loader):
    predictions = []
    references = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Print out the batch to understand its structure
            print(batch)  # Debugging line, remove once batch structure is clear
            
            # Convert inputs to tensors if they are lists
            input_ids = torch.tensor(batch["input_ids"]).to(device) if isinstance(batch["input_ids"], list) else batch["input_ids"].to(device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(device) if isinstance(batch["attention_mask"], list) else batch["attention_mask"].to(device)
            labels = torch.tensor(batch["labels"]).to(device) if isinstance(batch["labels"], list) else batch["labels"].to(device)

            # Ensure that batch dimensions are correct
            input_ids = input_ids.view(-1, input_ids.size(-1))  # Flatten if necessary
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))  # Flatten if necessary
            labels = labels.view(-1, labels.size(-1))  # Flatten if necessary

            # Generate predictions
            outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=512)
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            predictions.extend(decoded_preds)
            references.extend(decoded_refs)
    
    return predictions, references



# Get predictions and references
predictions, references = generate_predictions(model, data_loader)

# Evaluate BLEU score
bleu_score = bleu_metric.compute(predictions=predictions, references=references)
print(f"BLEU score: {bleu_score['bleu']}")

# Evaluate ROUGE score
rouge_score = rouge_metric.compute(predictions=predictions, references=references)
print(f"ROUGE score: {rouge_score}")

# Evaluate Exact Match (EM)
em_score = exact_match(predictions, references)
print(f"Exact Match score: {em_score}")