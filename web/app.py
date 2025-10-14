from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os, uuid, subprocess
from translate_speed_to_text import *
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

os.makedirs(TMP_DIR, exist_ok=True)

# @app.on_event("startup")
# def on_startup():
#     global conversation_chain
#     conversation_chain = load_model()

@app.get("/")
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    # 1. Lưu file tạm
    tmp_filename = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}.webm")
    with open(tmp_filename, "wb") as f:
        f.write(await file.read())

    # 2. Convert sang WAV
    wav_filename = tmp_filename.replace(".webm", ".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_filename, wav_filename],
            capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        return JSONResponse({"text": f"Lỗi convert: {e.stderr}"})
    
    text_output = process_audio(wav_filename)  # nếu muốn giữ tùy chọn denoise
    print(text_output)
    # text_output = get_response(conversation_chain, text_output)
    
    # 4. Xóa file tạm
    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)
    if os.path.exists(wav_filename):
        os.remove(wav_filename)

    return JSONResponse({"text": text_output})