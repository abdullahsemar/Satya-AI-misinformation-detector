# Deepfake Detector Extension

A full-stack application to detect deepfake images and videos directly in your browser using EfficientNet-B0.

## Architecture
- **Backend:** FastAPI server running a local PyTorch **EfficientNet-B0** model (`backend/models/efficientnet_b0.pt`).
- **Extension:** A Chrome manifest V3 extension that extracts images and video frames from web pages and sends them to the backend for analysis.

## Setup Instructions

### 1. Backend Setup
1. Open a terminal and navigate to `backend/`.
2. Install the necessary Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the backend server:
   ```bash
   python main.py
   ```
   *The server will start on `http://0.0.0.0:8000` using locally cached model weights.*

### 2. Extension Setup
1. Open Google Chrome and go to `chrome://extensions/`.
2. Enable **Developer mode**.
3. Click **Load unpacked** and select the `extension/` folder in this project.
4. The Deepfake Detector extension is now active!

### Usage
- Ensure the backend is running.
- Navigate to any web page containing images or videos.
- The extension will automatically scan and outline media elements.
- You can manually trigger a scan from the extension popup.
