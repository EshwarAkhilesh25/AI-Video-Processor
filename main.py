import os
import shutil
import subprocess
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
FRAMES_DIR = "frames"
PROCESSED_DIR = "processed_frames"
OUTPUT_DIR = "output"

for folder in [UPLOAD_DIR, FRAMES_DIR, PROCESSED_DIR, OUTPUT_DIR]:
    os.makedirs(folder, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def extract_frames(video_path):
    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vsync", "0",
        f"{FRAMES_DIR}/frame_%05d.png"
    ])

def get_video_fps(video_path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "0",
            "-of", "csv=p=0",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            video_path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    fps_fraction = result.stdout.strip()
    num, den = fps_fraction.split('/')
    return float(num) / float(den)

def reconstruct_video(video_path):
    final_video = os.path.join(OUTPUT_DIR, "final_output.mp4")

    fps = get_video_fps(video_path)

    subprocess.run([
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", f"{PROCESSED_DIR}/frame_%05d.png",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "16",
        "-pix_fmt", "yuv420p",
        final_video
    ])

    return final_video


def add_clean_white_outline(image):
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        img_np,
        contours,
        -1,
        (255, 255, 255),
        thickness=4  
    )

    return Image.fromarray(img_np)

def add_doodles_everywhere(image, colors):
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size

    def draw_star(x, y, color):
        size = random.randint(10, 20)
        draw.line((x-size, y, x+size, y), fill=color, width=3)
        draw.line((x, y-size, x, y+size), fill=color, width=3)
        draw.line((x-size, y-size, x+size, y+size), fill=color, width=2)
        draw.line((x-size, y+size, x+size, y-size), fill=color, width=2)

    def draw_zigzag(x, y, color):
        points = []
        for i in range(6):
            points.append((x + i*10, y + random.randint(-20,20)))
        draw.line(points, fill=color, width=3)

    def draw_spiral(x, y, color):
        radius = random.randint(15, 35)
        draw.arc(
            (x-radius, y-radius, x+radius, y+radius),
            start=0,
            end=random.randint(250, 320),
            fill=color,
            width=3
        )

    def draw_arrow(x, y, color):
        dx = random.randint(-50, 50)
        dy = random.randint(-50, 50)
        draw.line((x, y, x+dx, y+dy), fill=color, width=3)
        draw.line((x+dx, y+dy, x+dx-8, y+dy-8), fill=color, width=3)
        draw.line((x+dx, y+dy, x+dx+8, y+dy-8), fill=color, width=3)

    def draw_scribble_patch(x, y, color):
        for _ in range(6):
            draw.line(
                (
                    x + random.randint(-20,20),
                    y + random.randint(-20,20),
                    x + random.randint(-20,20),
                    y + random.randint(-20,20)
                ),
                fill=color,
                width=3
            )

    for _ in range(120):
        x = random.randint(0, width)
        y = random.randint(0, height)

        color = random.choice(colors)
        pattern = random.choice(
            ["line", "zigzag", "star", "spiral", "arrow", "scribble"]
        )

        if pattern == "line":
            draw.line(
                (x, y, x + random.randint(-80,80), y + random.randint(-80,80)),
                fill=color,
                width=4
            )
        elif pattern == "zigzag":
            draw_zigzag(x, y, color)
        elif pattern == "star":
            draw_star(x, y, color)
        elif pattern == "spiral":
            draw_spiral(x, y, color)
        elif pattern == "arrow":
            draw_arrow(x, y, color)
        elif pattern == "scribble":
            draw_scribble_patch(x, y, color)

    return image

@app.post("/process")
async def process_video(file: UploadFile = File(...)):

    try:
        for folder in [FRAMES_DIR, PROCESSED_DIR]:
            for f in os.listdir(folder):
                os.remove(os.path.join(folder, f))

        video_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extract_frames(video_path)

        frame_files = sorted(
            [f for f in os.listdir(FRAMES_DIR) if f.endswith(".png")]
        )

        if not frame_files:
            return {"error": "No frames extracted"}

        colors = ["red", "blue", "green", "yellow", "orange"]

        scribble_cache = None
        frames_per_update = 6

        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(FRAMES_DIR, frame_file)
            image = Image.open(frame_path).convert("RGB")

            if i % frames_per_update == 0 or scribble_cache is None:
                scribble_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
                scribble_layer = add_doodles_everywhere(scribble_layer, colors)
                scribble_cache = scribble_layer

            image = add_clean_white_outline(image)

            base_rgba = image.convert("RGBA")
            combined = Image.alpha_composite(base_rgba, scribble_cache)
            final_frame = combined.convert("RGB")

            final_frame.save(os.path.join(PROCESSED_DIR, frame_file))

        final_video = reconstruct_video(video_path)

        return FileResponse(
            final_video,
            media_type="video/mp4",
            filename="output.mp4"
        )

    except Exception as e:
        return {"error": str(e)}
