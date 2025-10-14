import os
import subprocess
from transformers import WhisperForConditionalGeneration, WhisperProcessor,  WhisperTokenizer, WhisperProcessor, WhisperFeatureExtractor
from peft import PeftModel
import sys

MODEL_REPO = "baduyne/whisper-small-vi"
MERGED_MODEL_PATH = "./whisper-finetune-small-vi"

# ==== 1. Merge LoRA ====
if not os.path.exists(MERGED_MODEL_PATH):
    print("Merging LoRA into base model...")

    # Load base model
    model_base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    # Load tokenizer & feature extractor từ model gốc openai
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-small", language="vi", task="transcribe"
    )
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Merge LoRA vào base
    model = PeftModel.from_pretrained(model_base, MODEL_REPO)
    model = model.merge_and_unload()

    # Lưu model + processor
    model.save_pretrained(MERGED_MODEL_PATH)
    processor.save_pretrained(MERGED_MODEL_PATH)
    print(f"Model merged and saved at: {MERGED_MODEL_PATH}")
else:
    print(f"Merged model already exists at: {MERGED_MODEL_PATH}")


# ==== 2. Cài đặt cmake và clone whisper.cpp ====
if not os.path.exists("whisper.cpp"):
    print("Installing dependencies and cloning whisper.cpp...")
    subprocess.run(["sudo", "apt", "update"])
    subprocess.run(["sudo", "apt", "install", "-y", "cmake", "build-essential", "python3-dev"])
else:
    print("whisper.cpp already exists, skipping clone.")


# ==== 3. Build whisper.cpp ====
os.chdir("whisper.cpp")

print("Configuring and building whisper.cpp with CMake...")
subprocess.run(["cmake", "-B", "build"], check=True)
subprocess.run(["cmake", "--build", "build", "-j", "--config", "Release"], check=True)

os.chdir("../")
# ==== 4. Convert sang GGML ====
GGML_PATH = "./ggml-model.bin"
if not os.path.exists(GGML_PATH):
    print("Converting Hugging Face model to GGML...")
    subprocess.run([
    "python3", "./whisper.cpp/models/convert-h5-to-ggml.py",
    MERGED_MODEL_PATH,        # thư mục model Hugging Face
    "./whisper",          # đường dẫn repo whisper.cpp
    "."                       # thư mục output cho file ggml
    ], check=True)

else:
    print(f" GGML model already exists: {GGML_PATH}")


# ==== 5. Quantize ====
Q5_PATH = "./ggml-model-q5.ggml"
if not os.path.exists(Q5_PATH):
    print(" Quantizing model to q5...")
    quantize_path = "./whisper.cpp/build/bin/quantize"
    subprocess.run([quantize_path, "./ggml-model.bin", Q5_PATH,"q5_1"])

else:
    print(f"Quantized model already exists: {Q5_PATH}")


# download model ollama 
subprocess.run(["ollama", "pull", "gemma3:1b"]) 