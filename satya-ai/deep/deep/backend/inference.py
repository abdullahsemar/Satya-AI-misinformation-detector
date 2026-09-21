import time
import numpy as np
from model import detector, classify_score
from preprocess import prepare_image

def run_inference(image_bytes: bytes) -> dict:
    """
    Executes end-to-end deepfake detection on raw image bytes using
    the dual Hugging Face model ensemble (prithivMLmods + umm-maybe).
    """
    start_time = time.perf_counter()

    # Preprocessing
    prep_start = time.perf_counter()
    image = prepare_image(image_bytes)
    prep_time_ms = round((time.perf_counter() - prep_start) * 1000.0, 2)

    # Ensemble Prediction
    result = detector.predict_image(image)
    
    total_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    result["preprocessing_time_ms"] = prep_time_ms
    result["total_inference_time_ms"] = total_time_ms

    return result

def run_video_inference(frames_bytes_list: list) -> dict:
    """
    Executes video frame detection using the dual-model ensemble with early exit logic.
    - Limits analysis to 3-5 frames max.
    - Early exit rule: If 3 consecutive frames have score >= 0.85, return HIGH-CONFIDENCE DEEPFAKE immediately.
    """
    start_time = time.perf_counter()
    
    # Cap frames to 5 maximum
    selected_frames = frames_bytes_list[:5]
    frame_scores = []
    consecutive_deepfake_frames = 0
    early_exit_triggered = False
    
    print(f"[VIDEO ENSEMBLE] Starting video analysis on {len(selected_frames)} selected frames...")
    
    for idx, frame_bytes in enumerate(selected_frames):
        try:
            image = prepare_image(frame_bytes)
            frame_res = detector.predict_image(image)
            frame_score = frame_res.get("deepfake_probability", 0.0)
            frame_scores.append(frame_score)
            
            print(f"[VIDEO ENSEMBLE] Frame {idx+1}/{len(selected_frames)} Score: {frame_score:.4f} (Consecutive Deepfakes: {consecutive_deepfake_frames + (1 if frame_score >= 0.85 else 0)})")
            
            # Consecutive Deepfake Logic:
            # If frame_score >= 0.85 -> consecutiveDeepfakeFrames += 1
            # Else -> consecutiveDeepfakeFrames = 0
            if frame_score >= 0.85:
                consecutive_deepfake_frames += 1
            else:
                consecutive_deepfake_frames = 0
                
            # Early exit condition: 3 consecutive deepfake frames
            if consecutive_deepfake_frames >= 3:
                early_exit_triggered = True
                print(f"[VIDEO ENSEMBLE] Early Exit Triggered! 3 consecutive deepfake frames detected.")
                break
                
        except Exception as e:
            print(f"[VIDEO ENSEMBLE] Warning: Frame {idx+1} processing error ({str(e)}). Skipping frame.")
            continue

    if not frame_scores:
        raise ValueError("Failed to process any valid video frames")

    # Early exit verdict or aggregated verdict
    if early_exit_triggered:
        final_video_score = round(float(np.mean(frame_scores[-3:])), 4)
        classification = classify_score(final_video_score)
        classification["reason"] = "High-confidence deepfake detected based on 3 consecutive deepfake frames (Early exit applied)."
    else:
        # Aggregated score using median for robust frame aggregation
        final_video_score = round(float(np.median(frame_scores)), 4)
        classification = classify_score(final_video_score)

    total_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    return {
        "media_type": "video",
        "verdict": classification["verdict"],
        "deepfake_probability": classification["deepfake_probability"],
        "deepfake_percentage": classification["deepfake_percentage"],
        "authenticity_probability": classification["authenticity_probability"],
        "authenticity_percentage": classification["authenticity_percentage"],
        "risk_level": classification["risk_level"],
        "reason": classification["reason"],
        "confidence": classification["deepfake_probability"],
        "is_deepfake": classification["is_deepfake"],
        "frames_analyzed": len(frame_scores),
        "early_exit": early_exit_triggered,
        "consecutive_deepfake_frames": consecutive_deepfake_frames,
        "frame_scores": frame_scores,
        "total_processing_time_ms": total_time_ms
    }
