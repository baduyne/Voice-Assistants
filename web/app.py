from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import json
import subprocess
import hashlib

# Import pipeline modules
from translate_speed_to_text import process_audio   # STT
from get_answers import get_response
from tts import text_to_speech                      # TTS

app = FastAPI()

# Mount frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Directories
AUDIO_DIR = "recordings"
TMP_DIR = "./tmp"
TTS_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(TTS_DIR, exist_ok=True)

tts_model_name = "./models/TTS_model.onnx"


@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    file_path = os.path.join(AUDIO_DIR, f"{client_id}.webm")

    # Xóa file cũ nếu tồn tại
    if os.path.exists(file_path):
        os.remove(file_path)

    try:
        while True:
            msg = await websocket.receive()

            # Nếu là bytes (chunk audio)
            if "bytes" in msg and msg["bytes"]:
                with open(file_path, "ab") as f:
                    f.write(msg["bytes"])

            # Nếu là text (control signal)
            elif "text" in msg and msg["text"]:
                data = json.loads(msg["text"])

                if data.get("type") == "finish":
                    # Convert sang WAV
                    wav_filename = os.path.join(TMP_DIR, f"{client_id}.wav")
                    try:
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", file_path, "-ar", "16000", "-ac", "1", wav_filename],
                            capture_output=True, text=True, check=True
                        )
                    except subprocess.CalledProcessError as e:
                        await websocket.send_text(json.dumps({"error": f"Lỗi convert: {e.stderr}"}))
                        continue

                    # Speech → Text
                    text_output = process_audio(wav_filename)
                    print(" STT:", text_output)

                    # Lấy phản hồi (nếu có chatbot)
                    bot_reply = get_response(text_output)  # hoặc gọi get_answer(text_output)

                    # Text → Speech (TTS)
                    text_hash = hashlib.sha1((bot_reply + "fast").encode("utf-8")).hexdigest()
                    audio_path = text_to_speech(bot_reply, "fast", tts_model_name, text_hash)

                    # Trả kết quả về client
                    tts_url = f"/audio/{os.path.basename(audio_path)}"
                    await websocket.send_text(json.dumps({
                        "status": "done",
                        "text": bot_reply,
                        "tts_url": tts_url
                    }))

                    # 6️⃣ Xóa file tạm
                    if os.path.exists(wav_filename):
                        os.remove(wav_filename)
                    if os.path.exists(file_path):
                        os.remove(file_path)

    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected")


@app.get("/audio/{filename}")
async def play_tts_audio(filename: str):
    file_path = os.path.join(TTS_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    return {"error": "File not found"}
