import runpod
import cv2
import numpy as np
import base64

def process_frame(job):
    job_input = job.get("input", {})
    frame_b64 = job_input.get("frame")

    if not frame_b64:
        return {"error": "No frame payload provided"}

    # Decode incoming base64 camera frame
    raw_bytes = base64.b64decode(frame_b64.split(",")[-1])
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Re-encode frame back to base64
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_output = base64.b64encode(buffer).decode('utf-8')

    return {"transformed_frame": f"data:image/jpeg;base64,{encoded_output}"}

runpod.serverless.start({"handler": process_frame})
