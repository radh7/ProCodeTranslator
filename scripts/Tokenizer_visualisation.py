import json
from transformers import T5Tokenizer

# Load the dataset from JSON
with open("data/train.json", 'r') as file:
    dataset = json.load(file)

tokenizer = T5Tokenizer.from_pretrained('t5-small')
for entry in dataset[:2]:
    java_code = entry['java']
    csharp_code = entry['cs']
    print()
    print('='*100)

    # Tokenize Java code
    java_tokens = tokenizer.tokenize(java_code)
    print(f"Java Code Tokens: {java_tokens}")
    print()
    print("=-"*50)
    # Tokenize C# code
    csharp_tokens = tokenizer.tokenize(csharp_code)
    print(f"C# Code Tokens: {csharp_tokens}")
    print("=-"*50)
    
print()
print("="*100)
# Tokenize and print token IDs
for entry in dataset[:2]:
    java_code = entry['java']
    csharp_code = entry['cs']

    # Encode Java code to token IDs
    java_token_ids = tokenizer.encode(java_code)
    print(f"Java Code Token IDs: {java_token_ids}")
    print()
    print("=-"*50)
    # Encode C# code to token IDs
    csharp_token_ids = tokenizer.encode(csharp_code)
    print(f"C# Code Token IDs: {csharp_token_ids}")
    print("=-"*50)

print("="*100)
# Decode back to check detokenized output
for entry in dataset[:2]:
    java_code = entry['java']
    csharp_code = entry['cs']

    java_token_ids = tokenizer.encode(java_code)
    csharp_token_ids = tokenizer.encode(csharp_code)

    java_detokenized = tokenizer.decode(java_token_ids, skip_special_tokens=True)
    csharp_detokenized = tokenizer.decode(csharp_token_ids, skip_special_tokens=True)

    print(f"Detokenized Java Code: {java_detokenized}")
    print('=-'*50)
    print(f"Detokenized C# Code: {csharp_detokenized}")
    print("=-"*50)

print("="*80)