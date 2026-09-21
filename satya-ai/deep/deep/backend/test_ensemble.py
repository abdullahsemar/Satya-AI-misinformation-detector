import urllib.request
import io
import numpy as np
from PIL import Image
from model import detector, classify_score
from inference import run_inference, run_video_inference

print("==================================================")
print("  TESTING CENTRALIZED SINGLE SOURCE OF TRUTH PIPELINE")
print("==================================================")

# 1. Test All Boundary Threshold Cases
print("\n--- 1. Testing Centralized Classification Boundary Cases ---")
test_cases = [
    (0.20, "AUTHENTIC", "LOW", 20, 80),
    (0.59, "AUTHENTIC", "LOW", 59, 41),
    (0.60, "SUSPICIOUS", "MEDIUM", 60, 40),
    (0.69, "SUSPICIOUS", "MEDIUM", 69, 31),
    (0.84, "SUSPICIOUS", "MEDIUM", 84, 16),
    (0.85, "DEEPFAKE", "HIGH", 85, 15),
    (0.95, "DEEPFAKE", "HIGH", 95, 5),
]

for score, expected_verdict, expected_risk, exp_df_pct, exp_auth_pct in test_cases:
    res = classify_score(score)
    print(f"Score: {score:.2f} -> Verdict: {res['verdict']} | Risk: {res['risk_level']} | Deepfake: {res['deepfake_percentage']}% | Auth: {res['authenticity_percentage']}%")
    assert res["verdict"] == expected_verdict, f"Expected {expected_verdict}, got {res['verdict']} for score {score}"
    assert res["risk_level"] == expected_risk, f"Expected {expected_risk}, got {res['risk_level']} for score {score}"
    assert res["deepfake_percentage"] == exp_df_pct, f"Expected {exp_df_pct}%, got {res['deepfake_percentage']}%"
    assert res["authenticity_percentage"] == exp_auth_pct, f"Expected {exp_auth_pct}%, got {res['authenticity_percentage']}%"

print("[OK] All 7 boundary test cases passed successfully!")

# 2. Test Specific Requested Payload Examples (42%, 69%, 91%)
print("\n--- 2. Testing Specific Response Format Examples (42%, 69%, 91%) ---")
ex_42 = classify_score(0.42)
ex_69 = classify_score(0.69)
ex_91 = classify_score(0.91)

print("\nExample 42% Payload:")
print(ex_42)
assert ex_42["verdict"] == "AUTHENTIC" and ex_42["deepfake_percentage"] == 42 and ex_42["authenticity_percentage"] == 58 and ex_42["risk_level"] == "LOW"

print("\nExample 69% Payload:")
print(ex_69)
assert ex_69["verdict"] == "SUSPICIOUS" and ex_69["deepfake_percentage"] == 69 and ex_69["authenticity_percentage"] == 31 and ex_69["risk_level"] == "MEDIUM"

print("\nExample 91% Payload:")
print(ex_91)
assert ex_91["verdict"] == "DEEPFAKE" and ex_91["deepfake_percentage"] == 91 and ex_91["authenticity_percentage"] == 9 and ex_91["risk_level"] == "HIGH"

print("[OK] All requested payload format examples verified!")

# 3. Test End-to-End Image Inference
print("\n--- 3. Testing End-to-End Image Inference ---")
sample_url = "https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg"
req = urllib.request.Request(sample_url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response:
    img_bytes = response.read()

res = run_inference(img_bytes)
print("Image Inference Output JSON:")
print(res)

assert "verdict" in res, "Missing 'verdict'"
assert "deepfake_probability" in res, "Missing 'deepfake_probability'"
assert "deepfake_percentage" in res, "Missing 'deepfake_percentage'"
assert "authenticity_probability" in res, "Missing 'authenticity_probability'"
assert "authenticity_percentage" in res, "Missing 'authenticity_percentage'"
assert "risk_level" in res, "Missing 'risk_level'"
assert "reason" in res, "Missing 'reason'"
assert "model1_score" in res, "Missing 'model1_score'"
assert "model2_score" in res, "Missing 'model2_score'"
assert res["authenticity_percentage"] == (100 - res["deepfake_percentage"]), "Authenticity percentage mismatch"
print("[OK] End-to-End Image Inference schema & logic verified!")

# 4. Test Video Frame Early Exit
print("\n--- 4. Testing Video Frame Early Exit ---")
frames = []
for i in range(5):
    noise_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pil_img = Image.fromarray(noise_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    frames.append(buf.getvalue())

video_res = run_video_inference(frames)
print("Video Inference Output JSON:")
print(video_res)
assert video_res["media_type"] == "video", "Missing media_type"
assert "verdict" in video_res, "Missing verdict"
assert "deepfake_percentage" in video_res, "Missing deepfake_percentage"
assert "authenticity_percentage" in video_res, "Missing authenticity_percentage"
assert video_res["authenticity_percentage"] == (100 - video_res["deepfake_percentage"]), "Authenticity percentage mismatch"
print("[OK] Video detection pipeline verified!")

print("\n==================================================")
print("  ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
print("==================================================")
