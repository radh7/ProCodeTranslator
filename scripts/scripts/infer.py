import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import re

model_path = "./models/checkpoint-4635"

# Load config
with open("config.json") as f:
    config = json.load(f)

# Load tokenizer (from pretrained model name)
print("Loading tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(model_path)
print("Tokenizer loaded!")

# Load model from local saved directory
print("Loading model...")
model = T5ForConditionalGeneration.from_pretrained(model_path)
print("Model loaded!")

# Set device (MPS / CPU fallback)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

def translate_code(code_snippet):
    print()
    print("*"*80)
    print("Translating code snippet...")
    # Code translation function
    input_text = f"translate java to csharp: {code_snippet.strip()}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=config["max_input_length"]).to(device)
    
    # # Debugging: print the input tokens
    # print("Input tokens:", input_ids)
    # print("Decoded input:", tokenizer.decode(input_ids[0], skip_special_tokens=True))

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=config["max_output_length"],  # maybe 512 or 768?
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2  # optional, helps avoid repetition
        )

    translated_code = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Translated Output (before post-processing):", translated_code)

    # Post-processing to fix common issues
    translated_code = post_process_translation(translated_code)

    
    #print("Translated Output (after post-processing):", translated_code)
    return translated_code

import re

def post_process_translation(raw_code: str) -> str:
    code = raw_code

    # Fix common typos and normalize syntax
    replacements = {
        "System.out1": "System.Console",
        "System.Out1": "System.Console",
        "System.Console.Print": "System.Console.WriteLine",
        "System.Console.Println": "System.Console.WriteLine",
        "System.Console.WriteLineln": "System.Console.WriteLine",
        "printLn": "WriteLine",
        "println": "WriteLine",
        "[": "(",
        "]": ")",
        ";": ";\n",
        "}else": "\n} else",
    }

    for old, new in replacements.items():
        code = code.replace(old, new)

    # Fix structure using regex
    code = re.sub(r"(public\s+class\s+\w+)", r"\1\n{\n\t", code)
    code = re.sub(r"(public\s+static\s+void\s+Main\(string\[\]\s+args\))", r"\1\n{\n\t", code)
    code = re.sub(r"Main\(string\(\) args\)", r"Main(string[] args)", code)
    code = re.sub(r"(\bif\s*\(.*?\))", r"\1\n{\n\t", code)
    code = re.sub(r"\}\s*else\s*\{?", r"}\nelse\n{\n\t", code)
    code = re.sub(r"(\bclass\s+\w+)(public|static)", r"\1 \n{\n\t    public static", code)
    code = re.sub(r"public class (\w+)public", r"public class \1 \n{\n\t    public", code)

    # Add braces if missing
    code = re.sub(r"\bMain\(string\[\] args\)(?!\s*{)", r"Main(string[] args) \n{\n\t", code)
    code = re.sub(r"\)(?!\s*;|\s*{)", r") \n{\n\t", code)

    # Ensure semicolons after statements
    code = re.sub(r'(?<=[^\s{};])\n', ';\n', code)

    # Fix double opening class or method
    code = re.sub(r'(\bclass\s+\w+)\s*\{?\s*(\bpublic\s+static\s+void\s+Main)', r'\1 \n{\n\t    \2', code)

    # Close any open code blocks if count mismatches
    open_braces = code.count("{")
    close_braces = code.count("}")
    if open_braces > close_braces:
        code += "\n" + "}" * (open_braces - close_braces)
    
    return code


# Test run
if __name__ == "__main__":
    # Java code to C
    # # test case 1
    sample_code1 = """
    public class HelloWorld {
        public static void main(String[] args) {
            System.out.println("Hello, World!");
        }
    }
    """
    result1 = translate_code(sample_code1)
    print("Input:\n", sample_code1)
    print("\nTranslated:\n", result1)

    #Java code to C
    # # test case 2
    sample_code2 = """
    public class PrimeChecker {
        public static void main(String[] args) {
            int num = 29;
            boolean isPrime = isPrime(num);
            
            if (isPrime) {
                System.out.println(num + " is a prime number.");
            } 
            else {
                System.out.println(num + " is not a prime number.");
            }
        }
    }

    """
    result2 = translate_code(sample_code2)
    print("Input:\n", sample_code2)
    print("\nTranslated:\n", result2)
