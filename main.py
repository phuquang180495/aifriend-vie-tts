import io
import os
import soundfile as sf
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

# 🎯 TỐI ƯU HÓA BỘ NHỚ RAM CHO PUTER / RENDER 512MB
import torch
torch.set_num_threads(1)
torch.set_grad_enabled(False)

from kokoro_vietnamese import KokoroVietnamese

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo mô hình
tts = KokoroVietnamese(device="cpu", voice="manh_dung")

VOICES = [
    "manh_dung", "diem_trinh", "hung_thinh", "mai_linh", 
    "mai_loan", "my_yen", "ngoc_huyen", "phat_tai", 
    "thanh_dat", "thuc_trinh", "tuan_ngoc", "duc_an", "duc_duy"
]

@app.get("/")
def health_check():
    return {"status": "Kokoro TTS Server is Running"}

@app.post("/api/tts")
async def generate_tts(data: dict):
    text = data.get("text", "")
    voice = data.get("voice", "manh_dung")
    
    if not text.strip():
        return Response(status_code=400)
    
    tts.voice = voice if voice in VOICES else "manh_dung"
    
    # 🎯 Tắt hoàn toàn bộ nhớ đệm Gradient của PyTorch để giải phóng RAM
    with torch.no_grad():
        audio, _ = tts.synthesize(text)
    
    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format='WAV')
    buf.seek(0)
    
    return Response(content=buf.read(), media_type="audio/wav")
