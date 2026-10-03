import runpod
import cv2
import numpy as np
import base64

def process_frame(job):
    job_input = job.get("input", {})
    
    # Safely extract frame or image payload
    frame_b64 = job_input.get("frame") or job_input.get("image")
    
    if not frame_b64:
        return {"error": "No frame payload provided"}

    # Strip base64 header if present (e.g., data:image/jpeg;base64,)
    if "," in frame_b64:
        frame_b64 = frame_b64.split(",")[1]

    # Decode incoming base64 camera frame
    raw_bytes = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Failed to decode image"}

    # Re-encode frame back to base64
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_output = base64.b64encode(buffer).decode('utf-8')

    return {"transformed_frame": f"data:image/jpeg;base64,{encoded_output}"}

runpod.serverless.start({"handler": process_frame})
