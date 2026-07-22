import io
import gc
import soundfile as sf
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

# 🎯 TỐI ƯU HÓA BỘ NHỚ RAM DƯỚI 512MB
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

tts_model = None

# 🎯 LAZY LOADING: Chỉ nạp mô hình vào RAM khi bắt đầu gọi phát âm
def get_tts_model():
    global tts_model
    if tts_model is None:
        tts_model = KokoroVietnamese(device="cpu", voice="manh_dung")
        gc.collect()
    return tts_model

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
    
    model = get_tts_model()
    model.voice = voice if voice in VOICES else "manh_dung"
    
    with torch.no_grad():
        audio, _ = model.synthesize(text)
    
    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format='WAV')
    buf.seek(0)
    
    gc.collect() # Giải phóng ngay RAM sau khi tạo audio xong
    return Response(content=buf.read(), media_type="audio/wav")
