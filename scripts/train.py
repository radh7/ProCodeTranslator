from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
import torch
import json

# Load config
with open("config.json") as f:
    config = json.load(f)

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(config["model_name"])
model = T5ForConditionalGeneration.from_pretrained(config["model_name"])

# Load dataset (ensure it has keys: 'java' and 'cs')
dataset = load_dataset("code_x_glue_cc_code_to_code_trans", split={'train': 'train', 'validation': 'validation', 'test': 'test'})

# Show sample to confirm structure
#print("Sample entry:", dataset['train'][0])

# For quick training/dev: shrink dataset or use splits
dataset = dataset["train"].train_test_split(test_size=0.1)
dataset = DatasetDict({
    "train": dataset["train"],
    "validation": dataset["test"]
})

# Map language keys correctly
LANG_KEY_MAP = {
    "java": "java",
    "c#": "cs",
    "csharp": "cs"
}

# Preprocessing function
def preprocess(batch):
    # Extract input and target code lists
    src_texts = [
        f"translate {config['source_lang']} to {config['target_lang']}: {code}"
        for code in batch[LANG_KEY_MAP[config["source_lang"]]]
    ]
    tgt_texts = batch[LANG_KEY_MAP[config["target_lang"]]]

    # Tokenize the batch
    model_inputs = tokenizer(
        src_texts,
        max_length=config["max_input_length"],
        truncation=True,
        padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            tgt_texts,
            max_length=config["max_output_length"],
            truncation=True,
            padding="max_length"
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize entire dataset
tokenized_dataset = dataset.map(preprocess, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./models",
    eval_strategy="epoch", 
    learning_rate=config["learning_rate"],
    per_device_train_batch_size=config["batch_size"],
    num_train_epochs=config["num_train_epochs"],
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
)

print('Training the model...')
# Start training
trainer.train()
print('Training complete!')
