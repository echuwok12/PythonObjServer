import os
from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import base64
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI()

# Allow CORS for WebSocket communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = YOLO(r"./yolov8n.pt")

# COCO class names for YOLOv8
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

@app.websocket("/detect-websocket")
async def detect_objects_ws(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connection established!")

    try:
        while True:
            # Receive Base64 encoded image from the client
            data = await websocket.receive_text()
            image_bytes = base64.b64decode(data)

            # Convert to OpenCV image
            image = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            height, width, _ = image.shape

            # Perform YOLOv8 object detection
            results = model(image)

            # Perform YOLOv8 object detection
            results = model(image)

            detections = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "Unknown"

                    detections.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": float(box.conf),
                        "bounding_box": [
                            float(box.xyxy[0][0]) / width,
                            float(box.xyxy[0][1]) / height,
                            float(box.xyxy[0][2]) / width,
                            float(box.xyxy[0][3]) / height
                        ]

                    })

            # Send detections back to the client
            await websocket.send_json({"detections": detections})

    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        await websocket.close()


from fastapi.responses import HTMLResponse

@app.get("/upload-video", response_class=HTMLResponse)
async def upload_video_form():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Upload Video for Detection</title>
    </head>
    <body>
        <h2>Upload Video for Object Detection</h2>
        <form id="uploadForm" enctype="multipart/form-data">
            <input type="file" name="file" accept="video/*" required>
            <button type="submit">Upload</button>
        </form>
        <pre id="output"></pre>
        <a id="downloadLink" style="display:none;" download="detection_results.json">Download Results</a>

        <script>
    const form = document.getElementById('uploadForm');
    const output = document.getElementById('output');
    const downloadLink = document.getElementById('downloadLink');
    const videoLink = document.createElement('a');
    videoLink.style.display = 'none';
    videoLink.textContent = '⬇️ Download Annotated Video';
    document.body.appendChild(videoLink);

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        output.textContent = "⏳ Uploading and processing...";
        downloadLink.style.display = 'none';
        videoLink.style.display = 'none';

        const formData = new FormData(form);
        const response = await fetch('/upload-video', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        output.textContent = JSON.stringify(result.results, null, 2);

        // JSON download link
        const blob = new Blob([JSON.stringify(result.results, null, 2)], { type: 'application/json' });
        downloadLink.href = URL.createObjectURL(blob);
        downloadLink.style.display = 'inline-block';
        downloadLink.textContent = '⬇️ Download Detection Results';

        // Video download link
        videoLink.href = result.video_url;
        videoLink.style.display = 'inline-block';
    });
</script>

    </body>
    </html>
    """
from fastapi.responses import FileResponse

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Endpoint to upload a video and return object detections + annotated video."""
    try:
        # Create upload and output folders
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)

        # Save input video
        contents = await file.read()
        input_path = f"uploads/{file.filename}"
        with open(input_path, "wb") as f:
            f.write(contents)

        # Setup video input/output
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        output_path = f"outputs/annotated_{file.filename}"
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        results_summary = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 1 == 0:  # process every frame
                results = model(frame)
                frame_detections = []

                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "Unknown"
                        conf = float(box.conf)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Draw box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        frame_detections.append({
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": conf,
                            "bounding_box": [x1, y1, x2, y2]
                        })

                results_summary.append({
                    "frame": frame_count,
                    "detections": frame_detections
                })

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()

        return JSONResponse(content={
            "results": results_summary,
            "video_url": f"/download/{os.path.basename(output_path)}"
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.get("/download/{filename}")
async def download_video(filename: str):
    file_path = f"outputs/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='video/mp4', filename=filename)
    return JSONResponse(status_code=404, content={"error": "File not found"})