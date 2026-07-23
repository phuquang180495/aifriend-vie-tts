import io
import gc
import soundfile as sf
from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 🎯 KHỐNG CHẾ BỘ NHỚ PYTORCH DƯỚI 200MB RAM
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)

from kokoro_vietnamese import KokoroVietnamese

app = FastAPI()

# 🎯 MỞ KHÓA CORS HOÀN TOÀN TỪ MỌI ORIGIN
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tts_model = None

VOICES = [
    "manh_dung", "diem_trinh", "hung_thinh", "mai_linh", 
    "mai_loan", "my_yen", "ngoc_huyen", "phat_tai", 
    "thanh_dat", "thuc_trinh", "tuan_ngoc", "duc_an", "duc_duy"
]

def get_model():
    global tts_model
    if tts_model is None:
        print("🚀 Đang khởi tạo mô hình Kokoro Vietnamese...")
        tts_model = KokoroVietnamese(device="cpu", voice="manh_dung")
        gc.collect()
        print("✅ Đã tải xong mô hình Kokoro!")
    return tts_model

@app.get("/")
def health_check():
    # Render sử dụng endpoint này để đánh thức server
    return {"status": "ok", "ready": tts_model is not None}

@app.post("/api/tts")
async def generate_tts(data: dict):
    try:
        model = get_model()
    except Exception as e:
        raise HTTPException(status_code=503, detail="Server đang khởi động mô hình")

    text = data.get("text", "").strip()
    voice = data.get("voice", "manh_dung")
    
    if not text:
        return Response(status_code=400)
    
    # Giới hạn an toàn 150 ký tự/chunk giúp CPU xử lý siêu tốc
    safe_text = text[:150]
    model.voice = voice if voice in VOICES else "manh_dung"
    
    try:
        with torch.inference_mode():
            audio, _ = model.synthesize(safe_text)
        
        buf = io.BytesIO()
        sf.write(buf, audio, 24000, format='WAV')
        buf.seek(0)
        
        gc.collect()
        return Response(content=buf.read(), media_type="audio/wav")
    except Exception as err:
        gc.collect()
        raise HTTPException(status_code=500, detail=str(err))
