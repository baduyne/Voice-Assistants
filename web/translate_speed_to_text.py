import subprocess
import os
import uuid

TMP_DIR = "./tmp_audio"
MODEL_PATH = "./models/ggml-base.en.bin"
os.makedirs(TMP_DIR, exist_ok=True)


def process_audio(file_path: str) -> str:

    ext = os.path.splitext(file_path)[1].lower()
    if ext != ".wav":
        wav_filename = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}.wav")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", file_path, wav_filename],
                capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError as e:
            return f"Error with convert audio: {e.stderr}"
    else:
        wav_filename = file_path
    txt_output = ""
    try:
        subprocess.run(
            [
                "../whisper.cpp/build/bin/whisper-cli",
                "-m", MODEL_PATH,
                "-f", wav_filename,
                "-l", "vi",
                "-otxt"
            ],
            capture_output=True, text=True, check=True
        )
        txt_file = wav_filename +  ".txt"
        if os.path.exists(txt_file):
            with open(txt_file, "r", encoding="utf-8") as f:
                txt_output = f.read().strip()
            os.remove(txt_file)
    except subprocess.CalledProcessError as e:
        txt_output = f"Whisper error: {e.stderr}"

    return txt_output
