import runpod
import cv2
import numpy as np
import base64
import os
import requests
import insightface
from insightface.app import FaceAnalysis

# Initialize InsightFace Analysis and Swapper models on GPU startup
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

# Path for swapper model
swapper_path = 'inswapper_128.onnx'

# Auto-download inswapper_128.onnx weights if not present locally
if not os.path.exists(swapper_path):
    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx"
    response = requests.get(url, stream=True)
    with open(swapper_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

swapper = insightface.model_zoo.get_model(swapper_path, download=False, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

def decode_b64(b64_str):
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    raw_bytes = base64.b64decode(b64_str)
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def process_frame(job):
    job_input = job.get("input", {})
    frame_b64 = job_input.get("frame") or job_input.get("image")
    target_face_b64 = job_input.get("target_image") or job_input.get("character_image")

    if not frame_b64:
        return {"error": "No camera frame provided"}

    # Decode incoming webcam frame
    source_img = decode_b64(frame_b64)
    if source_img is None:
        return {"error": "Failed to decode camera frame"}

    # Detect faces in live source webcam frame
    source_faces = app.get(source_img)
    if not source_faces:
        # Fallback to direct frame if no face is detected in current frame
        _, buffer = cv2.imencode('.jpg', source_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return {"transformed_frame": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"}

    source_face = source_faces[0]
    res_img = source_img.copy()

    # If target character picture is provided, perform neural face swap
    if target_face_b64:
        target_img = decode_b64(target_face_b64)
        if target_img is not None:
            target_faces = app.get(target_img)
            if target_faces:
                target_face = target_faces[0]
                # Execute GPU swap: paste target face onto live webcam pose
                res_img = swapper.get(res_img, source_face, target_face, paste_back=True)

    # Re-encode output to optimized JPEG
    _, buffer = cv2.imencode('.jpg', res_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    encoded_output = base64.b64encode(buffer).decode('utf-8')

    return {"transformed_frame": f"data:image/jpeg;base64,{encoded_output}"}

runpod.serverless.start({"handler": process_frame})
