from flask import Flask, request, jsonify, send_from_directory
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import json
import os
from infer import translate_code

app = Flask(__name__, static_folder="../frontend", static_url_path="")

# Load config
with open("config.json") as f:
    config = json.load(f)

model_path = "https://drive.google.com/drive/folders/1SqN3aTw2wJ6ZtslKfBzh2bCjZ4sLUn2o?usp=sharing"

# Debugging: Check if the model path exists
if not os.path.exists(model_path):
    print(f"[DEBUG] Model path does not exist: {model_path}")

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
print(f"[DEBUG] Using device: {device}")

@app.route("/")
def index():
    print("[DEBUG] Index route accessed")
    return send_from_directory("../frontend", "index.html")

@app.route("/translate", methods=["POST"])
def translate():
    print("[DEBUG] /translate route accessed")
    
    # Receive data
    try:
        data = request.get_json()
        print(f"[DEBUG] Received data: {data}")
    except Exception as e:
        print(f"[ERROR] Error parsing request data: {str(e)}")
        return jsonify({"error": "Failed to parse request data."}), 400
    
    java_code = data.get("code", "")
    
    if not java_code.strip():
        print("[DEBUG] No input Java code provided")
        return jsonify({"translated_code": "⚠️ No input provided."})
    
    try:
        print("[DEBUG] Translating Java code to C#...")
        translated_code = translate_code(java_code)
        print(f"[DEBUG] Translated code: {translated_code}")
        return jsonify({"translated_code": translated_code})
    except Exception as e:
        print(f"[ERROR] Error during translation: {str(e)}")
        return jsonify({"error": str(e), "translated_code": ""}), 500

if __name__ == "__main__":
    print("[DEBUG] Starting Flask app...")
    app.run(debug=True)  # Enable Flask's debug mode for detailed error logs
