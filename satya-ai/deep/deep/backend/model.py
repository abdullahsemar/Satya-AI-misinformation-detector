import time
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from utils import get_device, generate_xai_markers

MODEL_1_NAME = "prithivMLmods/Deep-Fake-Detector-Model"
MODEL_2_NAME = "umm-maybe/AI-image-detector"

def classify_score(deepfake_prob: float) -> dict:
    """
    Centralized Single Source of Truth Classification System.
    
    Thresholds:
      0% – 59% (0.00 - 0.5999):  AUTHENTIC  (LOW Risk)
      60% – 84% (0.60 - 0.8499): SUSPICIOUS (MEDIUM Risk)
      85% – 100% (0.85 - 1.00):  DEEPFAKE   (HIGH Risk)
    """
    score = max(0.0, min(1.0, float(deepfake_prob)))
    deepfake_pct = int(round(score * 100))
    auth_pct = 100 - deepfake_pct
    auth_prob = round(1.0 - score, 4)

    if score >= 0.85:
        verdict = "DEEPFAKE"
        risk_level = "HIGH"
        reason = "Both models detected strong evidence of AI-generated or manipulated content."
        is_deepfake = True
    elif score >= 0.60:
        verdict = "SUSPICIOUS"
        risk_level = "MEDIUM"
        reason = "The ensemble detected moderate evidence of AI-generated or manipulated content."
        is_deepfake = False
    else:
        verdict = "AUTHENTIC"
        risk_level = "LOW"
        reason = "Low evidence of AI-generated or manipulated content detected."
        is_deepfake = False

    return {
        "verdict": verdict,
        "deepfake_probability": round(score, 4),
        "deepfake_percentage": deepfake_pct,
        "authenticity_probability": auth_prob,
        "authenticity_percentage": auth_pct,
        "risk_level": risk_level,
        "reason": reason,
        "is_deepfake": is_deepfake
    }

class DualModelEnsembleDetector:
    """
    Dual-Model Ensemble AI Deepfake Detector combining:
    1. prithivMLmods/Deep-Fake-Detector-Model (Weight: 60%)
    2. umm-maybe/AI-image-detector (Weight: 40%)
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DualModelEnsembleDetector, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.device = get_device()
        print(f"Initializing Dual-Model Ensemble Detector on device: {self.device}...")
        
        # Model 1: prithivMLmods/Deep-Fake-Detector-Model {0: 'Fake', 1: 'Real'}
        print(f"Loading Model 1 ({MODEL_1_NAME})...")
        self.processor1 = AutoImageProcessor.from_pretrained(MODEL_1_NAME)
        self.model1 = AutoModelForImageClassification.from_pretrained(MODEL_1_NAME).to(self.device)
        self.model1.eval()

        # Model 2: umm-maybe/AI-image-detector {0: 'artificial', 1: 'human'}
        print(f"Loading Model 2 ({MODEL_2_NAME})...")
        self.processor2 = AutoImageProcessor.from_pretrained(MODEL_2_NAME)
        self.model2 = AutoModelForImageClassification.from_pretrained(MODEL_2_NAME).to(self.device)
        self.model2.eval()

        self._initialized = True
        print("Both Hugging Face Ensemble Models are Loaded and Ready!")

    def _predict_model1(self, image: Image.Image) -> dict:
        inputs = self.processor1(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model1(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).squeeze(0)
            
        return {
            "deepfake_probability": float(probs[0].item()),
            "real_probability": float(probs[1].item())
        }

    def _predict_model2(self, image: Image.Image) -> dict:
        inputs = self.processor2(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model2(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).squeeze(0)
            
        return {
            "deepfake_probability": float(probs[0].item()),
            "real_probability": float(probs[1].item())
        }

    def calculate_ensemble_score(self, m1_deepfake_prob: float, m2_deepfake_prob: float) -> dict:
        # Weighted combination: 60% Model 1 + 40% Model 2
        weighted_score = (0.60 * m1_deepfake_prob) + (0.40 * m2_deepfake_prob)

        # Agreement boost: +0.05 if both models independently > 0.70
        agreement_boost_applied = False
        if m1_deepfake_prob > 0.70 and m2_deepfake_prob > 0.70:
            weighted_score += 0.05
            agreement_boost_applied = True

        final_score = max(0.0, min(1.0, weighted_score))

        # Centralized classification
        classification = classify_score(final_score)
        classification["agreement_boost"] = agreement_boost_applied
        return classification

    def predict_image(self, image: Image.Image) -> dict:
        total_start = time.perf_counter()

        m1_start = time.perf_counter()
        m1_res = self._predict_model1(image)
        m1_time_ms = round((time.perf_counter() - m1_start) * 1000.0, 2)

        m2_start = time.perf_counter()
        m2_res = self._predict_model2(image)
        m2_time_ms = round((time.perf_counter() - m2_start) * 1000.0, 2)

        m1_score = round(m1_res["deepfake_probability"], 4)
        m2_score = round(m2_res["deepfake_probability"], 4)

        ensemble_res = self.calculate_ensemble_score(m1_res["deepfake_probability"], m2_res["deepfake_probability"])
        total_time_ms = round((time.perf_counter() - total_start) * 1000.0, 2)

        verdict = ensemble_res["verdict"]
        deepfake_prob = ensemble_res["deepfake_probability"]
        deepfake_pct = ensemble_res["deepfake_percentage"]

        print(f"[ML ENSEMBLE] Model 1: {m1_score:.4f} | Model 2: {m2_score:.4f} | Ensemble: {deepfake_prob:.4f} ({verdict})")

        xai_markers = generate_xai_markers(deepfake_prob, ensemble_res["is_deepfake"])

        details = (
            f"The media was classified as {verdict} because the ensemble detected a {deepfake_pct}% "
            f"probability of AI-generated or manipulated content. "
            f"Model 1 (Deep-Fake-Detector): {m1_score:.1%}, Model 2 (AI-image-detector): {m2_score:.1%}."
        )

        return {
            "verdict": verdict,
            "deepfake_probability": deepfake_prob,
            "deepfake_percentage": deepfake_pct,
            "authenticity_probability": ensemble_res["authenticity_probability"],
            "authenticity_percentage": ensemble_res["authenticity_percentage"],
            "risk_level": ensemble_res["risk_level"],
            "reason": ensemble_res["reason"],
            "confidence": deepfake_prob,
            "final_score": deepfake_prob,
            "model1_score": m1_score,
            "model2_score": m2_score,
            "is_deepfake": ensemble_res["is_deepfake"],
            "label": verdict.capitalize() if verdict != "DEEPFAKE" else "Deepfake",
            "details": details,
            "xai_markers": xai_markers,
            "processing_time_ms": total_time_ms,
            "model1_time_ms": m1_time_ms,
            "model2_time_ms": m2_time_ms
        }

# Global singleton detector instance
detector = DualModelEnsembleDetector()
