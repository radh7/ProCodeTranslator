import json
import torch
from datasets import load_dataset, Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from pathlib import Path

# Load config
with open("config.json") as f:
    config = json.load(f)
# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(config["model_name"])
model = T5ForConditionalGeneration.from_pretrained(config["model_name"])

# Load data from JSON
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def preprocess_function(example):
    prefix = f"translate {config['source_lang']} to {config['target_lang']}: "
    inputs = tokenizer(prefix + example["java"],
                       max_length=config["max_input_length"],
                       truncation=True,
                       padding="max_length")
    targets = tokenizer(example["cs"],
                        max_length=config["max_output_length"],
                        truncation=True,
                        padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

# Prepare datasets
train_data = Dataset.from_list(load_json("data/train.json"))
val_data = Dataset.from_list(load_json("data/validation.json"))

tokenized_train = train_data.map(preprocess_function, remove_columns=["java", "cs"])
tokenized_val = val_data.map(preprocess_function, remove_columns=["java", "cs"])

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./models/t5-finetuned",
    learning_rate=config["learning_rate"],
    per_device_train_batch_size=config["batch_size"],
    per_device_eval_batch_size=config["batch_size"],
    num_train_epochs=config["num_train_epochs"],
    predict_with_generate=True,
)

# Trainer setup
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

# Train!
trainer.train()

# Save final model
model.save_pretrained("./models/t5-finetuned")
tokenizer.save_pretrained("./models/t5-finetuned")
