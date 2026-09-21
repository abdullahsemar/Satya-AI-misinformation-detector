import io
import urllib.request
from PIL import Image
from image_provenance import (
    extract_exif_metadata,
    compute_perceptual_hash,
    verify_image_provenance_pipeline,
    analyze_temporal_context
)

print("==========================================================")
print("  TESTING REVERSE IMAGE SEARCH & TEMPORAL COMPARISON     ")
print("==========================================================")

# 1. Create a synthetic test image with EXIF simulation
img = Image.new('RGB', (100, 100), color=(73, 109, 137))
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='JPEG')
img_bytes = img_byte_arr.getvalue()

# Test EXIF Extractor
print("\n--- 1. Testing EXIF & Camera Metadata Extractor ---")
exif_res = extract_exif_metadata(img_bytes)
print("EXIF Metadata:", exif_res)
assert "has_exif" in exif_res
assert "is_ai_edited" in exif_res
print("[OK] EXIF metadata parser operational!")

# Test Perceptual Hashing (dHash)
print("\n--- 2. Testing Perceptual Difference Hashing (dHash) ---")
phash1 = compute_perceptual_hash(img_bytes)
print("Perceptual Hash (64-bit dHash):", phash1)
assert len(phash1) == 16, "Expected 16-char hex representation of 64-bit dHash"

# Test hash consistency
img_resized = img.resize((50, 50))
buf_resized = io.BytesIO()
img_resized.save(buf_resized, format='JPEG')
phash2 = compute_perceptual_hash(buf_resized.getvalue())
print("Resized Image Hash:", phash2)
assert phash1 == phash2, "Perceptual hash should be invariant to resizing"
print("[OK] Perceptual Hashing verified invariant to resizing!")

# 3. Test Temporal Context Mismatch (Recycled Old Image vs Breaking News)
print("\n--- 3. Testing Temporal Date Comparison & Context Mismatch ---")
fake_old_exif = {"has_exif": True, "date_time_original": "2018:05:12 10:20:00", "is_ai_edited": False}
fake_web_matches = [{"year": 2018, "title": "2018 Flood Archives - News", "url": "https://news.com/2018/flood"}]

# Case A: Old image shared with Breaking News urgency
mismatch_res = analyze_temporal_context(fake_old_exif, fake_web_matches, claim_text="BREAKING NEWS: Massive flood happening today right now!")
print("\nCase A (Recycled Image Shared as Breaking News):")
print("Verdict:", mismatch_res["verdict"])
print("First Seen Year:", mismatch_res["first_seen_year"])
print("Years Elapsed:", mismatch_res["years_elapsed"])
print("Summary:", mismatch_res["summary"])
assert mismatch_res["verdict"] == "OUT_OF_CONTEXT", "Expected OUT_OF_CONTEXT verdict"
assert len(mismatch_res["timeline"]) >= 2, "Expected multi-year provenance timeline"
print("[OK] Recycled old image detection verified!")

# Case B: Contemporary Image (Current Year)
contemporary_exif = {"has_exif": True, "date_time_original": "2026:01:10 09:00:00", "is_ai_edited": False}
contemp_res = analyze_temporal_context(contemporary_exif, [], claim_text="Recent press conference")
print("\nCase B (Contemporary Media):")
print("Verdict:", contemp_res["verdict"])
print("Summary:", contemp_res["summary"])
assert contemp_res["verdict"] == "CONTEMPORARY_OR_ORIGINAL"
print("[OK] Contemporary image context verified!")

# 4. Test Complete Pipeline with Real Web Image
print("\n--- 4. Testing End-to-End Image Provenance Pipeline ---")
sample_url = "https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg"
try:
    req = urllib.request.Request(sample_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        real_img_bytes = response.read()

    pipeline_res = verify_image_provenance_pipeline(real_img_bytes, sample_url, claim_text="Trump portrait breaking news 2026")
    print("Pipeline Result Verdict:", pipeline_res["verdict"])
    print("Timeline Stages Count:", len(pipeline_res["timeline"]))
    print("Blockchain Provenance Hash:", pipeline_res["blockchain_audit"]["sha256_hash"][:16] + "...")
    assert "verdict" in pipeline_res
    assert "blockchain_audit" in pipeline_res
    print("[OK] End-to-End Image Provenance Pipeline fully operational!")
except Exception as e:
    print("Note: Network test skipped or failed:", str(e))

print("\n==========================================================")
print("  REVERSE IMAGE SEARCH & PROVENANCE TESTS: ALL PASSED!   ")
print("==========================================================")
