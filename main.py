from fastapi import FastAPI, File, UploadFile, Request
from concurrent.futures import ProcessPoolExecutor
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import neural_network
import numpy as np
import imageio
import asyncio
import logging
import base64
import cv2


NUM_WORKERS = 4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("weapon_detection.log"),
        logging.StreamHandler()
    ]
)

file_handler = logging.FileHandler("weapon_detection.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

for name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
    uvicorn_logger = logging.getLogger(name)
    uvicorn_logger.setLevel(logging.INFO)
    uvicorn_logger.handlers.clear()
    uvicorn_logger.addHandler(file_handler)
    uvicorn_logger.propagate = False

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting...")

    app.state.process_pool = ProcessPoolExecutor(
        max_workers=NUM_WORKERS
    )

    asyncio.create_task(simulate_internal_task())

    yield

    try:
        app.state.process_pool.shutdown(wait=True)
        logger.info("Process pool shutdown")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


app = FastAPI(
    title="WeaponDetection API",
    lifespan=lifespan
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


async def simulate_internal_task():
    logger.info("Model compilation within worker processes started...")
    loop = asyncio.get_running_loop()

    tasks = [
        loop.run_in_executor(app.state.process_pool, neural_network.init_neural_model)
        for _ in range(0, NUM_WORKERS)
    ]

    results = await asyncio.gather(*tasks)


def render_result(request, message, image_data=None, video_data=None):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": message,
        "image_data": image_data,
        'video_data': video_data,
    })


async def read_video(file: UploadFile) -> list[np.ndarray]:
    video_bytes = await file.read()
    temp_path = f"uploads/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(video_bytes)
    cap = cv2.VideoCapture(temp_path)
    frames = []

    while True:
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)

    cap.release()
    return frames


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": None})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        if not (file.content_type.startswith("image/") or file.content_type.startswith("video/")) \
                or file.content_type in {"image/heic", "image/heif"}:
            logger.warning(f"Invalid file type uploaded: {file.content_type}")
            return render_result(request, 'Invalid file type uploaded')

        if file.content_type.startswith("image/"):
            img_bytes = await file.read()
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                logger.error("Uploaded image could not be decoded")
                return render_result(request, 'Uploaded object could not be decoded')
        elif file.content_type.startswith("video/"):
            img = await read_video(file)

        loop = asyncio.get_running_loop()
        pid, result, imgs, inf = await loop.run_in_executor(
            app.state.process_pool,
            neural_network.process_input,
            img,
            file.content_type
        )

        if result != "Error":
            logger.info(f"PID={pid} filename={file.filename} result: {result}")
            if file.content_type.startswith("image/"):
                imgs = imgs[0].plot()
                _, buffer = cv2.imencode('.jpg', imgs)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                return render_result(request, result, image_data=img_base64)
            if file.content_type.startswith("video/"):
                import tempfile
                frames = [cv2.cvtColor(img.plot(), cv2.COLOR_BGR2RGB) for img in imgs]
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    temp_path = tmp.name

                writer = imageio.get_writer(temp_path, format='ffmpeg', mode='I', fps=24, codec='libx264')
                for frame in frames:
                    writer.append_data(frame)
                writer.close()

                with open(temp_path, "rb") as f:
                    video_data = f.read()
                encoded = base64.b64encode(video_data).decode("utf-8")
                return render_result(request, result, video_data=encoded)
        else:
            logger.error(f"PID={pid} filename={file.filename} result: Error during detection")
            return render_result(request, 'Error during detection')

    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return render_result(request, 'Failed to process object')

