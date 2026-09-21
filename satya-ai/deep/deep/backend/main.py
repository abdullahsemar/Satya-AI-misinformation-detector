import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
from inference import run_inference, run_video_inference
from model import detector
from satya_engine import verify_text_pipeline, nli_engine

app = FastAPI(
    title="SATYA — AI Forward-Checker & Deepfake Detector API",
    description="High-performance backend combining Visual Deepfake Detection and Multimodal Text Verification"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class TextVerificationRequest(BaseModel):
    text: str
    target_lang: Optional[str] = "en"

@app.on_event("startup")
async def startup_event():
    """Ensures both HuggingFace ensemble models and the NLI model are preloaded exactly once on startup."""
    print("Preloading HuggingFace Ensemble Models during FastAPI startup...")
    _ = detector.device
    print("Preloading NLI Engine (cross-encoder/nli-MiniLM2-L6-H768)...")
    _ = nli_engine.device
    print("SATYA Multi-Model Engine startup initialization complete!")

@app.get("/")
@app.get("/health")
async def health():
    return {
        "status": "online",
        "service": "SATYA — AI Forward-Checker & Deepfake Detector Platform",
        "models": [
            "prithivMLmods/Deep-Fake-Detector-Model (60%)",
            "umm-maybe/AI-image-detector (40%)",
            "cross-encoder/nli-MiniLM2-L6-H768 (NLI Text Stance)"
        ],
        "version": "SATYA v2.5",
        "docs": "http://127.0.0.1:8000/docs",
        "predict_endpoint": "http://127.0.0.1:8000/predict",
        "verify_text_endpoint": "http://127.0.0.1:8000/verify-text"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty payload received")

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image payload exceeded 10MB limit")

        result = run_inference(image_bytes)
        result["input_type"] = "image"
        return result

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[API ERROR] Image inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/predict_video")
async def predict_video(files: List[UploadFile] = File(...)):
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No video frame files uploaded")
            
        frames_bytes = []
        for file in files[:5]: # Max 5 frames
            content = await file.read()
            if content:
                frames_bytes.append(content)
                
        if not frames_bytes:
            raise HTTPException(status_code=400, detail="All uploaded frame payloads were empty")

        result = run_video_inference(frames_bytes)
        result["input_type"] = "video"
        return result

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[API ERROR] Video inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video inference failed: {str(e)}")

@app.post("/verify-text")
async def verify_text(req: TextVerificationRequest):
    try:
        if not req.text or not req.text.strip():
            raise HTTPException(status_code=400, detail="Empty text input provided")

        result = verify_text_pipeline(req.text, req.target_lang or "en")
        return result
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[API ERROR] Text verification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text verification failed: {str(e)}")

@app.post("/analyze-image-provenance")
async def analyze_image_provenance_endpoint(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    claim_text: Optional[str] = Form(""),
    target_lang: Optional[str] = Form("en")
):
    try:
        image_bytes = None
        if file:
            image_bytes = await file.read()
        elif image_url and image_url.startswith("http"):
            try:
                resp = requests.get(image_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=3.0)
                if resp.status_code == 200:
                    image_bytes = resp.content
            except Exception:
                pass

        if not image_bytes:
            raise HTTPException(status_code=400, detail="Valid image file or reachable image_url required")

        from image_provenance import verify_image_provenance_pipeline
        result = verify_image_provenance_pipeline(
            image_bytes=image_bytes,
            image_url=image_url,
            claim_text=claim_text or "",
            target_lang=target_lang or "en"
        )
        return result
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[API ERROR] Image provenance analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image provenance analysis failed: {str(e)}")

@app.post("/verify")
async def verify_unified(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    target_lang: Optional[str] = Form("en")
):
    try:
        if file:
            content = await file.read()
            if content:
                res = run_inference(content)
                res["input_type"] = "image"
                return res

        if text and text.strip():
            res = verify_text_pipeline(text, target_lang or "en")
            res["input_type"] = "text"
            return res

        raise HTTPException(status_code=400, detail="Neither image file nor text payload provided")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unified verification failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
