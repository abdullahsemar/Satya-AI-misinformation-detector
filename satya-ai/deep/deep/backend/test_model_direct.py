import urllib.request
import io
import numpy as np
from PIL import Image
from model import detector

req = urllib.request.Request("https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg", headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response:
    image_bytes = response.read()

print("Testing EfficientNet-B0 Direct Inference (Sample Real Image):")
real_res = detector.predict(image_bytes)
print(real_res)

# Test noise image for synthetic test
noise_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
buf = io.BytesIO()
noise_img.save(buf, format="JPEG")
noise_bytes = buf.getvalue()

print("\nTesting EfficientNet-B0 Direct Inference (Synthetic Noise Image):")
noise_res = detector.predict(noise_bytes)
print(noise_res)
