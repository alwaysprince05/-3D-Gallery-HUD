from flask import Flask, render_template_string, Response
import cv2
import mediapipe as mp
import numpy as np
import os
import glob
from collections import deque

app = Flask(__name__)

# --- CONFIG ---
GALLERY_PATH = "gallery"
THUMB_SIZE = (180, 120)
NEON = (57, 255, 20)
NEON2 = (255, 255, 0)
BG_COLOR = (10, 15, 20)
HUD_ALPHA = 0.35
MAX_HISTORY = 8

# --- LOAD IMAGES ---
def load_images(path=GALLERY_PATH, size=THUMB_SIZE):
    images = []
    if not os.path.exists(path):
        return images
    for f in sorted(glob.glob(os.path.join(path, "*.jpg")) + glob.glob(os.path.join(path, "*.png"))):
        img = cv2.imread(f)
        if img is not None:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            images.append(img)
    return images

def draw_gallery(frame, images, angle, zoom, selected_idx, anim_t):
    if not images:
        cv2.putText(frame, "No Images in gallery/", (150, 240), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 2)
        return frame
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2 + 30
    n = len(images)
    radius = int(220 * zoom)
    depth = 180 * zoom
    img_w, img_h = THUMB_SIZE
    order = []
    for i in range(n):
        theta = (2 * np.pi * i / n) + angle
        x = int(cx + radius * np.cos(theta))
        y = int(cy + radius * np.sin(theta) * 0.5)
        z = int(depth * np.sin(theta))
        scale = 1 + 0.25 * (z / depth)
        img = cv2.resize(images[i], (int(img_w * scale), int(img_h * scale)))
        ix, iy = x - img.shape[1] // 2, y - img.shape[0] // 2
        order.append((z, img, ix, iy, i))
    order.sort(key=lambda tup: tup[0])
    for z, img, ix, iy, idx in order:
        ih, iw = img.shape[:2]
        x0, y0 = max(ix, 0), max(iy, 0)
        x1, y1 = min(ix+iw, w), min(iy+ih, h)
        img_x0, img_y0 = x0 - ix, y0 - iy
        img_x1, img_y1 = iw - (ix+iw-x1), ih - (iy+ih-y1)
        if x0 < x1 and y0 < y1:
            border = 4 if idx == selected_idx else 2
            color = NEON2 if idx == selected_idx else NEON
            cv2.rectangle(frame, (x0-6, y0-6), (x1+6, y1+6), color, border, cv2.LINE_AA)
            roi = frame[y0:y1, x0:x1]
            img_roi = img[img_y0:img_y1, img_x0:img_x1]
            if roi.shape == img_roi.shape:
                frame[y0:y1, x0:x1] = cv2.addWeighted(roi, 0.3, img_roi, 0.7, 0)
    return frame

def detect_gesture(hand_landmarks, history):
    if not hand_landmarks:
        return None, None, None, None, None
    lm = hand_landmarks[0].landmark
    def pt(idx): return np.array([lm[idx].x, lm[idx].y])
    thumb_tip = pt(4)
    index_tip = pt(8)
    center = (index_tip + thumb_tip) / 2
    pinch_dist = np.linalg.norm(thumb_tip - index_tip)
    history.append(center)
    if len(history) > 1:
        move = history[-1] - history[-2]
    else:
        move = np.array([0, 0])
    return pinch_dist, move, center, thumb_tip, index_tip

def draw_hud(frame, hand_landmarks, gesture, anim_t):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), NEON, -1)
    cv2.addWeighted(overlay, HUD_ALPHA, frame, 1-HUD_ALPHA, 0, frame)
    cv2.putText(frame, "Gesture-Controlled 3D Gallery HUD", (18, 28), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,0), 2, cv2.LINE_AA)
    cv2.drawMarker(frame, (w//2, h//2+30), NEON, markerType=cv2.MARKER_CROSS, markerSize=32, thickness=2)
    scan_y = int((np.sin(anim_t*2) * 0.5 + 0.5) * (h-80)) + 40
    cv2.line(frame, (0, scan_y), (w, scan_y), NEON2, 2)
    if hand_landmarks and gesture:
        for hand in hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand, mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=NEON, thickness=2, circle_radius=4),
                mp.solutions.drawing_utils.DrawingSpec(color=NEON2, thickness=3, circle_radius=2)
            )
        thumb_tip, index_tip = gesture[3], gesture[4]
        if thumb_tip is not None and index_tip is not None:
            tx, ty = int(thumb_tip[0] * w), int(thumb_tip[1] * h)
            ix, iy = int(index_tip[0] * w), int(index_tip[1] * h)
            cx, cy = int((thumb_tip[0]+index_tip[0])/2 * w), int((thumb_tip[1]+index_tip[1])/2 * h)
            cv2.circle(frame, (cx, cy), 12, NEON2, -1)
            cv2.line(frame, (tx, ty), (ix, iy), NEON2, 3)
            cv2.putText(frame, "Pinch: Zoom | Move: Rotate", (ix+20, iy-20), cv2.FONT_HERSHEY_PLAIN, 1.5, NEON2, 2, cv2.LINE_AA)
    return frame

def generate_frames():
    cap = cv2.VideoCapture(0)
    images = load_images()
    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)
    
    angle = 0.0
    zoom = 1.0
    selected_idx = 0
    anim_t = 0
    move_history = deque(maxlen=MAX_HISTORY)
    prev_center = None
    base_pinch = None
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            # Fallback frame if camera is unavailable
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for Camera...", (150, 240), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
            hand_landmarks = None
            pinch_dist, move, center, thumb_tip, index_tip = None, np.array([0,0]), None, None, None
        else:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            hand_landmarks = results.multi_hand_landmarks
            
            pinch_dist, move, center, thumb_tip, index_tip = None, np.array([0,0]), None, None, None
            if hand_landmarks:
                pinch_dist, move, center, thumb_tip, index_tip = detect_gesture(hand_landmarks, move_history)
                if base_pinch is None and pinch_dist is not None:
                    base_pinch = pinch_dist
                if base_pinch is not None and pinch_dist is not None:
                    zoom = np.clip(1.0 + (pinch_dist - base_pinch) * 18.0, 0.3, 3.5)
                if prev_center is not None and center is not None:
                    dx = center[0] - prev_center[0]
                    angle += dx * 32.0
                prev_center = center
            
        hud_frame = frame.copy()
        hud_frame = draw_hud(hud_frame, hand_landmarks, (pinch_dist, move, center, thumb_tip, index_tip), anim_t)
        
        gallery_frame = np.full((480, 640, 3), BG_COLOR, dtype=np.uint8)
        gallery_frame = draw_gallery(gallery_frame, images, angle, zoom, selected_idx, anim_t)
        
        anim_t += 0.045
        
        combined = np.hstack((hud_frame, gallery_frame))
        ret, buffer = cv2.imencode('.jpg', combined)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    html = """
    <html>
        <head>
            <title>3D Gallery HUD</title>
            <style>
                body { background-color: #0a0f14; color: #39ff14; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; margin: 0; padding: 20px; }
                img { border: 2px solid #39ff14; border-radius: 8px; box-shadow: 0 0 20px #39ff14; max-width: 100%; margin-top: 20px; }
                h1 { text-shadow: 0 0 10px #39ff14; margin-bottom: 5px; }
                p { color: #fff; opacity: 0.8; }
            </style>
        </head>
        <body>
            <h1>Gesture-Controlled 3D Gallery HUD</h1>
            <p>Left: Webcam Tracker | Right: Interactive 3D Gallery</p>
            <img src="/video_feed" />
        </body>
    </html>
    """
    return render_template_string(html)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8765, debug=True)
