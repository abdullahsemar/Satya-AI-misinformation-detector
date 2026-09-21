import urllib.request
import io
from inference import run_inference

print("Testing EfficientNet-B0 Deepfake Detector pipeline...")
url = "https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

with urllib.request.urlopen(req) as response:
    image_bytes = response.read()

result = run_inference(image_bytes)
print("Inference Result:")
for key, value in result.items():
    print(f"  {key}: {value}")
