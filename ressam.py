import cv2
import mediapipe as mp
import numpy as np
import random
import math
import os
import time
import pygame
from datetime import datetime
from collections import deque

# ======================================
# 1. APP BRAND
# ======================================
APP_NAME = "SUDMOTION STUDIO"
APP_SUBTITLE = "GESTURE DRAW SYSTEM"

# ======================================
# 1.1 AUDIO
# ======================================
MUSIC_PATH = os.path.join("assets", "music", "background.mp3")
music_enabled = True
audio_ready = False
music_paused = False

# ======================================
# 2. MEDIA PIPE
# ======================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    max_num_hands=1
)
mp_draw = mp.solutions.drawing_utils

# ======================================
# 3. STATE
# ======================================
canvas = None
points = deque(maxlen=1024)
particles = []

smooth_x, smooth_y = None, None
SMOOTHING = 0.22

effect_mode = "spark"            # spark, magic, fire
brush_style = "silk"             # silk, ink, laser, smoke
background_mode = "gradient"     # camera, dark, gradient, spotlight
brush_base_thickness = 4

mirror_mode = False
kaleido_mode = False

fist_was_active = False
last_action_text = ""
last_action_timer = 0

SAVE_DIR = "captures"
VIDEO_DIR = "recordings"
EXPORT_DIR = "exports"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# recording
is_recording = False
video_writer = None
video_fps = 20.0
recording_start_time = None

# intro
show_intro = True
program_start_time = time.time()
intro_transition_active = False
intro_transition_start = None
INTRO_TRANSITION_DURATION = 0.55

# menu
menu_open = False
open_palm_latch = False
open_palm_counter = 0
OPEN_PALM_HOLD_FRAMES = 10
menu_selected_button = None

# hover select
hover_selected_button = None
hover_start_time = None
HOVER_SELECT_SECONDS = 1.0

# export
latest_clean_frame = None

# replay
current_stroke = []
stroke_history = []
is_replaying = False
replay_strokes = []
replay_stroke_index = 0
replay_point_index = 1
replay_canvas = None

# ======================================
# 4. COLORS / THEMES
# ======================================
COLOR_PALETTES = [
    {"name": "EMERALD",   "core": (110, 255, 170), "glow": (200, 255, 230)},
    {"name": "ICEBLUE",   "core": (255, 220, 120), "glow": (255, 245, 210)},
    {"name": "ROSENEON",  "core": (190, 130, 255), "glow": (235, 210, 255)},
    {"name": "AMBERGOLD", "core": (80, 200, 255),  "glow": (180, 235, 255)},
    {"name": "SOFTWHITE", "core": (245, 245, 245), "glow": (255, 255, 255)},
]
color_index = 0

BRUSH_STYLES = ["silk", "ink", "laser", "smoke"]
BACKGROUND_MODES = ["camera", "dark", "gradient", "spotlight"]

# 3D pixel UI colors (BGR)
UI_OUTER = (18, 12, 34)
UI_PANEL = (48, 28, 68)
UI_PANEL_2 = (66, 38, 92)
UI_HILITE = (170, 230, 255)
UI_SHADOW = (28, 10, 38)
UI_BORDER = (255, 214, 90)
UI_TEXT = (245, 245, 245)
UI_DIM = (190, 190, 205)
UI_ACCENT = (255, 255, 90)
UI_CYAN = (255, 245, 120)
UI_RED = (40, 40, 220)
UI_GREEN = (90, 220, 120)
UI_BLACK = (10, 10, 16)

# ======================================
# 4.1 MENU LAYOUT
# ======================================
MENU_PANEL_X1 = 14
MENU_PANEL_Y1 = 236
MENU_PANEL_X2 = 530
MENU_PANEL_Y2 = 500

MENU_INNER_PAD_X = 16
MENU_INNER_PAD_Y = 16

MENU_BUTTON_W = 140
MENU_BUTTON_H = 40

MENU_GAP_X = 16
MENU_GAP_Y = 16

MENU_STATUS_HEIGHT = 32
MENU_STATUS_GAP = 16

# ======================================
# 5. PARTICLE
# ======================================
class Particle:
    def __init__(self, x, y, mode="spark", burst=False):
        self.x = x
        self.y = y
        speed = 5 if burst else 2.2
        self.vx = random.uniform(-speed, speed)
        self.vy = random.uniform(-speed, speed)
        self.life = random.randint(170, 255) if burst else random.randint(90, 180)
        self.mode = mode
        self.radius = random.randint(2, 4) if burst else random.randint(1, 2)

        if mode == "spark":
            self.color = (
                random.randint(140, 255),
                255,
                random.randint(140, 220)
            )
        elif mode == "magic":
            self.color = (
                random.randint(210, 255),
                random.randint(80, 170),
                255
            )
        elif mode == "fire":
            self.color = (
                0,
                random.randint(110, 210),
                random.randint(220, 255)
            )
        else:
            self.color = (255, 255, 255)

    def move(self):
        self.x += self.vx
        self.y += self.vy

        if self.mode == "fire":
            self.y -= 0.35
            self.vx *= 0.985
            self.vy *= 0.985

        self.life -= 11

# ======================================
# 6. GESTURE HELPERS
# ======================================
def is_finger_up(hand_lms, tip_id, pip_id):
    return hand_lms.landmark[tip_id].y < hand_lms.landmark[pip_id].y

def is_fist(hand_lms):
    index_closed = hand_lms.landmark[8].y > hand_lms.landmark[6].y
    middle_closed = hand_lms.landmark[12].y > hand_lms.landmark[10].y
    ring_closed = hand_lms.landmark[16].y > hand_lms.landmark[14].y
    pinky_closed = hand_lms.landmark[20].y > hand_lms.landmark[18].y
    return index_closed and middle_closed and ring_closed and pinky_closed

def is_open_palm(hand_lms):
    index_up = is_finger_up(hand_lms, 8, 6)
    middle_up = is_finger_up(hand_lms, 12, 10)
    ring_up = is_finger_up(hand_lms, 16, 14)
    pinky_up = is_finger_up(hand_lms, 20, 18)
    return index_up and middle_up and ring_up and pinky_up

# ======================================
# 7. GENERAL HELPERS
# ======================================
def show_action(text, duration=30):
    global last_action_text, last_action_timer
    last_action_text = text
    last_action_timer = duration

def init_audio():
    global audio_ready, music_enabled, music_paused

    try:
        pygame.mixer.init()
        audio_ready = True
        music_paused = False

        if os.path.exists(MUSIC_PATH):
            try:
                pygame.mixer.music.load(MUSIC_PATH)
                pygame.mixer.music.set_volume(0.4)

                if music_enabled:
                    pygame.mixer.music.play(-1)

                show_action("MUSIC READY", duration=20)

            except Exception as e:
                print("Music load error:", e)
                audio_ready = False
                music_enabled = False
                music_paused = False
                show_action("MUSIC LOAD FAIL", duration=25)

        else:
            audio_ready = False
            music_enabled = False
            music_paused = False
            show_action("NO MUSIC FILE", duration=25)

    except Exception as e:
        print("Audio init error:", e)
        audio_ready = False
        music_enabled = False
        music_paused = False
        show_action("AUDIO DISABLED", duration=25)

def toggle_music():
    global music_enabled, music_paused

    if not audio_ready:
        show_action("AUDIO NOT READY", duration=20)
        return

    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.pause()
            music_enabled = False
            music_paused = True
            show_action("MUSIC OFF", duration=20)
        else:
            if music_paused:
                pygame.mixer.music.unpause()
            else:
                pygame.mixer.music.play(-1)

            pygame.mixer.music.set_volume(0.4)
            music_enabled = True
            music_paused = False
            show_action("MUSIC ON", duration=20)

    except Exception as e:
        print("Music toggle error:", e)
        show_action("MUSIC ERROR", duration=20)

def get_current_palette():
    return COLOR_PALETTES[color_index]

def clear_canvas(shape):
    return np.zeros(shape, dtype=np.uint8)

def cycle_color():
    global color_index
    color_index = (color_index + 1) % len(COLOR_PALETTES)
    show_action(f"COLOR {get_current_palette()['name']}", duration=20)

def cycle_brush_style():
    global brush_style
    idx = BRUSH_STYLES.index(brush_style)
    brush_style = BRUSH_STYLES[(idx + 1) % len(BRUSH_STYLES)]
    show_action(f"BRUSH {brush_style.upper()}", duration=20)

def cycle_background():
    global background_mode
    idx = BACKGROUND_MODES.index(background_mode)
    background_mode = BACKGROUND_MODES[(idx + 1) % len(BACKGROUND_MODES)]
    show_action(f"BG {background_mode.upper()}", duration=20)

def toggle_mirror():
    global mirror_mode
    mirror_mode = not mirror_mode
    show_action(f"MIRROR {'ON' if mirror_mode else 'OFF'}", duration=20)

def toggle_kaleido():
    global kaleido_mode
    kaleido_mode = not kaleido_mode
    show_action(f"KALEIDO {'ON' if kaleido_mode else 'OFF'}", duration=20)

def save_screenshot(img):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, f"sude_capture_{timestamp}.png")
    cv2.imwrite(path, img)
    show_action("SCREENSHOT SAVED", duration=30)

def format_seconds(seconds):
    total = max(0, int(seconds))
    minutes = total // 60
    secs = total % 60
    return f"{minutes:02d}:{secs:02d}"

def start_video_recording(frame_width, frame_height):
    global is_recording, video_writer, recording_start_time

    if is_recording:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(VIDEO_DIR, f"sude_record_{timestamp}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(path, fourcc, video_fps, (frame_width, frame_height))

    if not video_writer.isOpened():
        video_writer = None
        show_action("REC FAILED", duration=30)
        return

    is_recording = True
    recording_start_time = time.time()
    show_action("REC STARTED", duration=25)

def stop_video_recording():
    global is_recording, video_writer, recording_start_time

    if video_writer is not None:
        video_writer.release()
        video_writer = None

    is_recording = False
    recording_start_time = None
    show_action("REC STOPPED", duration=25)

def toggle_video_recording(frame_width, frame_height):
    if is_recording:
        stop_video_recording()
    else:
        start_video_recording(frame_width, frame_height)

def export_transparent_png(canvas_img, out_path):
    gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(canvas_img)
    rgba = cv2.merge([b, g, r, alpha])
    cv2.imwrite(out_path, rgba)

def export_session(clean_frame, canvas_img):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(EXPORT_DIR, f"sude_session_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)

    transparent_path = os.path.join(session_dir, "artwork_transparent.png")
    poster_path = os.path.join(session_dir, "poster_frame.png")
    info_path = os.path.join(session_dir, "session_info.txt")

    export_transparent_png(canvas_img, transparent_path)
    cv2.imwrite(poster_path, clean_frame)

    info_text = (
        f"PROJECT: {APP_NAME}\n"
        f"SUBTITLE: {APP_SUBTITLE}\n"
        f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"EFFECT MODE: {effect_mode}\n"
        f"BRUSH STYLE: {brush_style}\n"
        f"BACKGROUND: {background_mode}\n"
        f"COLOR: {get_current_palette()['name']}\n"
        f"BRUSH SIZE: {brush_base_thickness}\n"
        f"MIRROR MODE: {mirror_mode}\n"
        f"KALEIDO MODE: {kaleido_mode}\n"
        f"RECORDING ACTIVE: {is_recording}\n"
        f"MUSIC ACTIVE: {music_enabled}\n"
        f"STROKE COUNT: {len(stroke_history)}\n"
    )

    with open(info_path, "w", encoding="utf-8") as f:
        f.write(info_text)

    show_action("SESSION EXPORTED", duration=30)

def draw_pointer_glow(img, x, y, palette):
    glow = np.zeros_like(img)
    cv2.circle(glow, (x, y), 24, palette["glow"], -1, lineType=cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), 10)
    img[:] = cv2.addWeighted(img, 1.0, glow, 0.16, 0)
    cv2.circle(img, (x, y), 8, palette["core"], 2, lineType=cv2.LINE_AA)
    cv2.circle(img, (x, y), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)

def add_draw_sparks(overlay_frame, x, y, palette):
    for _ in range(3):
        dx = random.randint(-8, 8)
        dy = random.randint(-8, 8)
        r = random.randint(1, 2)
        cv2.circle(
            overlay_frame,
            (x + dx, y + dy),
            r,
            palette["glow"],
            -1,
            lineType=cv2.LINE_AA
        )

def apply_intro_transition(frame, transition_start, duration):
    elapsed = time.time() - transition_start
    alpha = min(1.0, elapsed / duration)

    zoom = 1.18 - 0.18 * alpha
    h, w = frame.shape[:2]
    zw = int(w * zoom)
    zh = int(h * zoom)

    resized = cv2.resize(frame, (zw, zh))
    x1 = max(0, (zw - w) // 2)
    y1 = max(0, (zh - h) // 2)
    cropped = resized[y1:y1 + h, x1:x1 + w]

    if cropped.shape[0] != h or cropped.shape[1] != w:
        cropped = cv2.resize(cropped, (w, h))

    fade = np.zeros_like(cropped)
    fade_amount = max(0.0, 0.22 * (1.0 - alpha))
    out = cv2.addWeighted(cropped, 1.0 - fade_amount, fade, fade_amount, 0)
    done = alpha >= 1.0
    return out, done

# ======================================
# 8. BACKGROUNDS
# ======================================
def apply_vignette(img, strength=0.45):
    h, w = img.shape[:2]
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    xv, yv = np.meshgrid(x, y)
    dist = np.sqrt(xv * xv + yv * yv)
    mask = 1 - np.clip((dist - 0.2) / 1.0, 0, 1)
    mask = mask ** 1.7
    mask = (1 - strength) + strength * mask
    vignette = np.dstack([mask, mask, mask])
    out = img.astype(np.float32) * vignette
    return np.clip(out, 0, 255).astype(np.uint8)

def create_gradient_background(shape):
    h, w, _ = shape
    bg = np.zeros(shape, dtype=np.uint8)

    top = np.array([36, 18, 74], dtype=np.float32)
    mid = np.array([20, 12, 46], dtype=np.float32)
    bottom = np.array([6, 6, 16], dtype=np.float32)

    for y in range(h):
        t = y / max(1, h - 1)
        if t < 0.5:
            t2 = t / 0.5
            color = (1 - t2) * top + t2 * mid
        else:
            t2 = (t - 0.5) / 0.5
            color = (1 - t2) * mid + t2 * bottom
        bg[y, :, :] = color

    return apply_vignette(bg, 0.50)

def create_spotlight_background(shape):
    h, w, _ = shape
    bg = np.zeros(shape, dtype=np.uint8)
    bg[:] = (10, 10, 18)

    center_x, center_y = w // 2, h // 2
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    mask = np.exp(-(dist ** 2) / (2 * (min(h, w) * 0.22) ** 2))
    mask = np.clip(mask, 0, 1)

    bg[:, :, 0] = bg[:, :, 0] + (mask * 52).astype(np.uint8)
    bg[:, :, 1] = bg[:, :, 1] + (mask * 32).astype(np.uint8)
    bg[:, :, 2] = bg[:, :, 2] + (mask * 22).astype(np.uint8)

    return apply_vignette(bg, 0.55)

def get_base_frame(camera_img):
    if background_mode == "camera":
        return camera_img.copy()
    if background_mode == "dark":
        return np.zeros_like(camera_img)
    if background_mode == "gradient":
        return create_gradient_background(camera_img.shape)
    if background_mode == "spotlight":
        return create_spotlight_background(camera_img.shape)
    return camera_img.copy()

# ======================================
# 9. BRUSHES
# ======================================
def draw_silk_brush(canvas_img, p1, p2, speed_value, brush_size, core, glow):
    thickness = int(max(2, min(8, brush_size + 2 - speed_value * 0.08)))
    temp = np.zeros_like(canvas_img)
    cv2.line(temp, p1, p2, glow, thickness + 8, lineType=cv2.LINE_AA)
    cv2.line(temp, p1, p2, glow, thickness + 4, lineType=cv2.LINE_AA)
    blurred = cv2.GaussianBlur(temp, (0, 0), 4)
    canvas_img[:] = cv2.addWeighted(canvas_img, 1.0, blurred, 0.28, 0)

    mid = np.zeros_like(canvas_img)
    cv2.line(mid, p1, p2, glow, thickness + 2, lineType=cv2.LINE_AA)
    mid = cv2.GaussianBlur(mid, (0, 0), 1.8)
    canvas_img[:] = cv2.addWeighted(canvas_img, 1.0, mid, 0.42, 0)

    cv2.line(canvas_img, p1, p2, core, thickness, lineType=cv2.LINE_AA)

def draw_ink_brush(canvas_img, p1, p2, speed_value, brush_size, core, glow):
    thickness = int(max(1, min(7, brush_size + 1 - speed_value * 0.05)))
    dark_core = tuple(max(0, c - 60) for c in core)
    cv2.line(canvas_img, p1, p2, dark_core, thickness + 2, lineType=cv2.LINE_AA)
    cv2.line(canvas_img, p1, p2, core, thickness, lineType=cv2.LINE_AA)

    if random.random() < 0.35:
        jitter = random.randint(-2, 2)
        p1j = (p1[0] + jitter, p1[1] + jitter)
        p2j = (p2[0] - jitter, p2[1] - jitter)
        cv2.line(canvas_img, p1j, p2j, glow, max(1, thickness - 1), lineType=cv2.LINE_AA)

def draw_laser_brush(canvas_img, p1, p2, speed_value, brush_size, core, glow):
    thickness = int(max(2, min(6, brush_size)))
    temp = np.zeros_like(canvas_img)
    cv2.line(temp, p1, p2, glow, thickness + 10, lineType=cv2.LINE_AA)
    temp = cv2.GaussianBlur(temp, (0, 0), 6)
    canvas_img[:] = cv2.addWeighted(canvas_img, 1.0, temp, 0.22, 0)
    cv2.line(canvas_img, p1, p2, glow, thickness + 2, lineType=cv2.LINE_AA)
    cv2.line(canvas_img, p1, p2, core, thickness, lineType=cv2.LINE_AA)
    cv2.line(canvas_img, p1, p2, (255, 255, 255), max(1, thickness - 2), lineType=cv2.LINE_AA)

def draw_smoke_brush(canvas_img, p1, p2, speed_value, brush_size, core, glow):
    thickness = int(max(3, min(10, brush_size + 2)))
    temp = np.zeros_like(canvas_img)
    smoke_color = tuple(int((c + g) / 2) for c, g in zip(core, glow))
    cv2.line(temp, p1, p2, smoke_color, thickness + 10, lineType=cv2.LINE_AA)
    temp = cv2.GaussianBlur(temp, (0, 0), 8)
    canvas_img[:] = cv2.addWeighted(canvas_img, 1.0, temp, 0.16, 0)
    cv2.line(canvas_img, p1, p2, glow, thickness, lineType=cv2.LINE_AA)
    soft = cv2.GaussianBlur(canvas_img, (0, 0), 0.35)
    canvas_img[:] = cv2.addWeighted(canvas_img, 0.86, soft, 0.14, 0)

def draw_basic_brush(canvas_img, p1, p2, speed_value, brush_size):
    palette = get_current_palette()
    core = palette["core"]
    glow = palette["glow"]

    if brush_style == "silk":
        draw_silk_brush(canvas_img, p1, p2, speed_value, brush_size, core, glow)
    elif brush_style == "ink":
        draw_ink_brush(canvas_img, p1, p2, speed_value, brush_size, core, glow)
    elif brush_style == "laser":
        draw_laser_brush(canvas_img, p1, p2, speed_value, brush_size, core, glow)
    elif brush_style == "smoke":
        draw_smoke_brush(canvas_img, p1, p2, speed_value, brush_size, core, glow)

def draw_brush_line(canvas_img, p1, p2, speed_value, brush_size, frame_w, frame_h):
    if kaleido_mode:
        points_set = [
            ((p1[0], p1[1]), (p2[0], p2[1])),
            ((frame_w - p1[0], p1[1]), (frame_w - p2[0], p2[1])),
            ((p1[0], frame_h - p1[1]), (p2[0], frame_h - p2[1])),
            ((frame_w - p1[0], frame_h - p1[1]), (frame_w - p2[0], frame_h - p2[1]))
        ]
        for a, b in points_set:
            draw_basic_brush(canvas_img, a, b, speed_value, brush_size)
    else:
        draw_basic_brush(canvas_img, p1, p2, speed_value, brush_size)
        if mirror_mode:
            mp1 = (frame_w - p1[0], p1[1])
            mp2 = (frame_w - p2[0], p2[1])
            draw_basic_brush(canvas_img, mp1, mp2, speed_value, brush_size)

# ======================================
# 10. 3D PIXEL UI
# ======================================
def draw_pixel_panel_3d(img, x1, y1, x2, y2, body=UI_PANEL, border=UI_BORDER):
    cv2.rectangle(img, (x1, y1), (x2, y2), border, -1)
    cv2.rectangle(img, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), UI_OUTER, -1)
    cv2.rectangle(img, (x1 + 6, y1 + 6), (x2 - 6, y2 - 6), body, -1)

    cv2.line(img, (x1 + 6, y1 + 6), (x2 - 6, y1 + 6), UI_HILITE, 2)
    cv2.line(img, (x1 + 6, y1 + 6), (x1 + 6, y2 - 6), UI_HILITE, 2)
    cv2.line(img, (x1 + 6, y2 - 6), (x2 - 6, y2 - 6), UI_SHADOW, 3)
    cv2.line(img, (x2 - 6, y1 + 6), (x2 - 6, y2 - 6), UI_SHADOW, 3)

    cv2.rectangle(img, (x1 + 10, y1 + 10), (x2 - 10, y2 - 10), UI_PANEL_2, 1)

def draw_pixel_button_3d(img, x1, y1, x2, y2, selected=False, progress=0.0):
    body = (70, 48, 96) if not selected else (100, 78, 128)
    border = UI_BORDER if selected else (170, 150, 210)
    draw_pixel_panel_3d(img, x1, y1, x2, y2, body=body, border=border)

    if selected:
        cv2.rectangle(img, (x1 + 12, y1 + 12), (x2 - 12, y2 - 12), UI_ACCENT, 1)

    if progress > 0:
        bar_margin = 10
        bar_h = 6
        bx1 = x1 + bar_margin
        bx2 = x2 - bar_margin
        by1 = y2 - 13
        by2 = by1 + bar_h

        cv2.rectangle(img, (bx1, by1), (bx2, by2), UI_OUTER, -1)
        fill_w = int((bx2 - bx1) * min(1.0, max(0.0, progress)))
        if fill_w > 0:
            cv2.rectangle(img, (bx1, by1), (bx1 + fill_w, by2), UI_ACCENT, -1)

def draw_recording_badge(img):
    if not is_recording:
        return

    elapsed = 0 if recording_start_time is None else time.time() - recording_start_time
    timer_text = format_seconds(elapsed)

    x1, y1, x2, y2 = img.shape[1] - 215, 18, img.shape[1] - 18, 62
    draw_pixel_panel_3d(img, x1, y1, x2, y2, body=(60, 20, 34), border=UI_BORDER)

    blink = int(time.time() * 2) % 2 == 0
    dot_color = (0, 0, 255) if blink else (40, 40, 120)
    cv2.circle(img, (x1 + 28, y1 + 22), 8, dot_color, -1, lineType=cv2.LINE_AA)

    cv2.putText(img, "REC", (x1 + 45, y1 + 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, UI_TEXT, 2, lineType=cv2.LINE_AA)
    cv2.putText(img, timer_text, (x1 + 105, y1 + 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, UI_TEXT, 2, lineType=cv2.LINE_AA)

def draw_hud(img, effect_mode, brush_size):
    draw_pixel_panel_3d(img, 16, 14, 920, 190, body=(48, 26, 74), border=UI_BORDER)

    cv2.putText(img, APP_NAME, (34, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.90, UI_TEXT, 3, lineType=cv2.LINE_AA)

    cv2.putText(img, APP_SUBTITLE, (36, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, UI_DIM, 1, lineType=cv2.LINE_AA)

    cv2.putText(img, f"EFFECT : {effect_mode.upper()}",
                (36, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.56, UI_TEXT, 1, lineType=cv2.LINE_AA)
    cv2.putText(img, f"BRUSH  : {brush_style.upper()}",
                (240, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.56, UI_TEXT, 1, lineType=cv2.LINE_AA)
    cv2.putText(img, f"BG     : {background_mode.upper()}",
                (450, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.56, UI_TEXT, 1, lineType=cv2.LINE_AA)

    cv2.putText(img, f"COLOR  : {get_current_palette()['name']}",
                (36, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.56, UI_TEXT, 1, lineType=cv2.LINE_AA)
    cv2.putText(img, f"SIZE   : {brush_size}",
                (240, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.56, UI_TEXT, 1, lineType=cv2.LINE_AA)
    cv2.putText(img, f"MENU   : {'OPEN' if menu_open else 'CLOSED'}",
                (450, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.56, UI_TEXT, 1, lineType=cv2.LINE_AA)

    cv2.putText(img, f"MIRROR : {'ON' if mirror_mode else 'OFF'}",
                (36, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.52, UI_CYAN, 1, lineType=cv2.LINE_AA)
    cv2.putText(img, f"KALEIDO: {'ON' if kaleido_mode else 'OFF'}",
                (240, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.52, UI_CYAN, 1, lineType=cv2.LINE_AA)
    cv2.putText(img, f"MUSIC  : {'ON' if music_enabled else 'OFF'}",
                (450, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.52, UI_CYAN, 1, lineType=cv2.LINE_AA)

def draw_status_text(img):
    global last_action_timer, last_action_text
    if last_action_timer > 0:
        draw_pixel_panel_3d(img, 22, 202, 470, 246, body=(58, 26, 42), border=UI_BORDER)
        cv2.putText(img, last_action_text, (38, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, UI_TEXT, 2, lineType=cv2.LINE_AA)
        last_action_timer -= 1

def get_menu_icon(label):
    icons = {
        "COLOR": "C",
        "BRUSH": "B",
        "BG": "G",
        "REC": "R",
        "REPLAY": "P",
        "CLEAR": "X",
        "EXPORT": "E",
        "MIRROR": "M",
        "KALEIDO": "K",
    }
    return icons.get(label, "?")

def get_gesture_menu_buttons():
    labels = ["COLOR", "BRUSH", "BG", "REC", "REPLAY", "CLEAR", "EXPORT", "MIRROR", "KALEIDO"]

    start_x = MENU_PANEL_X1 + MENU_INNER_PAD_X
    start_y = MENU_PANEL_Y1 + MENU_INNER_PAD_Y

    buttons = []
    for i, label in enumerate(labels):
        col = i % 3
        row = i // 3

        x1 = start_x + col * (MENU_BUTTON_W + MENU_GAP_X)
        y1 = start_y + row * (MENU_BUTTON_H + MENU_GAP_Y)
        x2 = x1 + MENU_BUTTON_W
        y2 = y1 + MENU_BUTTON_H

        buttons.append({
            "name": label,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

    return buttons

def get_status_bar_rect():
    start_x = MENU_PANEL_X1 + MENU_INNER_PAD_X
    start_y = MENU_PANEL_Y1 + MENU_INNER_PAD_Y

    grid_height = (3 * MENU_BUTTON_H) + (2 * MENU_GAP_Y)

    x1 = start_x
    x2 = MENU_PANEL_X2 - MENU_INNER_PAD_X
    y1 = start_y + grid_height + MENU_STATUS_GAP
    y2 = y1 + MENU_STATUS_HEIGHT

    return x1, y1, x2, y2

def point_in_rect(px, py, rect):
    return rect["x1"] <= px <= rect["x2"] and rect["y1"] <= py <= rect["y2"]

def get_hovered_button(px, py):
    for btn in get_gesture_menu_buttons():
        if point_in_rect(px, py, btn):
            return btn["name"]
    return None

def execute_menu_action(name, frame_w, frame_h):
    global canvas, points, current_stroke, stroke_history
    global replay_canvas, is_replaying, latest_clean_frame, menu_open
    global smooth_x, smooth_y
    global hover_selected_button, hover_start_time

    if name == "COLOR":
        cycle_color()
    elif name == "BRUSH":
        cycle_brush_style()
    elif name == "BG":
        cycle_background()
    elif name == "REC":
        toggle_video_recording(frame_w, frame_h)
    elif name == "REPLAY":
        start_replay((frame_h, frame_w, 3))
    elif name == "CLEAR":
        canvas = clear_canvas((frame_h, frame_w, 3))
        points.clear()
        current_stroke = []
        stroke_history = []
        replay_canvas = None
        is_replaying = False
        show_action("ALL CLEARED", duration=20)
    elif name == "EXPORT":
        if latest_clean_frame is not None:
            export_session(latest_clean_frame, canvas)
    elif name == "MIRROR":
        toggle_mirror()
    elif name == "KALEIDO":
        toggle_kaleido()

    menu_open = False
    points.clear()
    smooth_x, smooth_y = None, None
    hover_selected_button = None
    hover_start_time = None
    show_action(f"{name} SELECTED", duration=18)

def get_hover_progress(button_name):
    if button_name is None:
        return 0.0
    if hover_selected_button != button_name:
        return 0.0
    if hover_start_time is None:
        return 0.0
    elapsed = time.time() - hover_start_time
    return min(1.0, elapsed / HOVER_SELECT_SECONDS)

def draw_gesture_menu(img, fingertip=None, selected_name=None):
    draw_pixel_panel_3d(
        img,
        MENU_PANEL_X1, MENU_PANEL_Y1, MENU_PANEL_X2, MENU_PANEL_Y2,
        body=(42, 24, 70), border=UI_BORDER
    )

    if fingertip is not None:
        fx, fy = fingertip
        cv2.rectangle(img, (fx - 8, fy - 8), (fx + 8, fy + 8), UI_ACCENT, 2)
        cv2.rectangle(img, (fx - 3, fy - 3), (fx + 3, fy + 3), UI_TEXT, -1)

    for btn in get_gesture_menu_buttons():
        is_selected = (btn["name"] == selected_name)
        progress = get_hover_progress(btn["name"])

        draw_pixel_button_3d(
            img,
            btn["x1"], btn["y1"], btn["x2"], btn["y2"],
            selected=is_selected,
            progress=progress
        )

        ix1, iy1 = btn["x1"] + 8, btn["y1"] + 7
        ix2, iy2 = ix1 + 24, iy1 + 24

        icon_body = (86, 56, 118) if not is_selected else (0, 230, 240)
        icon_border = (190, 190, 230) if not is_selected else UI_BORDER
        draw_pixel_panel_3d(img, ix1, iy1, ix2, iy2, body=icon_body, border=icon_border)

        icon_color = UI_TEXT if not is_selected else UI_BLACK
        cv2.putText(img, get_menu_icon(btn["name"]), (ix1 + 5, iy1 + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, icon_color, 2, lineType=cv2.LINE_AA)

        cv2.putText(img, btn["name"], (btn["x1"] + 40, btn["y1"] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, UI_TEXT, 1, lineType=cv2.LINE_AA)

        if is_selected:
            cv2.putText(img, ">", (btn["x2"] - 15, btn["y1"] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, UI_ACCENT, 2, lineType=cv2.LINE_AA)

    status_x1, status_y1, status_x2, status_y2 = get_status_bar_rect()
    draw_pixel_panel_3d(img, status_x1, status_y1, status_x2, status_y2,
                        body=(56, 34, 86), border=UI_BORDER)

    hint = f"SELECTED: {selected_name if selected_name else 'NONE'}"
    cv2.putText(img, hint, (status_x1 + 12, status_y1 + 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, UI_TEXT, 1, lineType=cv2.LINE_AA)

# ======================================
# 10.1 INTRO FX
# ======================================
def draw_intro_particles(img, x1, y1, x2, y2, t):
    perimeter = 2 * ((x2 - x1) + (y2 - y1))
    if perimeter <= 0:
        return

    particle_count = 18

    for i in range(particle_count):
        offset = (t * 90 + i * (perimeter / particle_count)) % perimeter

        if offset < (x2 - x1):
            px = x1 + int(offset)
            py = y1
        elif offset < (x2 - x1) + (y2 - y1):
            px = x2
            py = y1 + int(offset - (x2 - x1))
        elif offset < 2 * (x2 - x1) + (y2 - y1):
            px = x2 - int(offset - ((x2 - x1) + (y2 - y1)))
            py = y2
        else:
            px = x1
            py = y2 - int(offset - (2 * (x2 - x1) + (y2 - y1)))

        drift = int(3 + 2 * math.sin(t * 1.8 + i))

        if py == y1:
            py -= drift
        elif px == x2:
            px += drift
        elif py == y2:
            py += drift
        elif px == x1:
            px -= drift

        r = 2 + int((math.sin(t * 2.5 + i) + 1) * 1.2)

        color = (
            170 + int(40 * math.sin(i + t)),
            190 + int(35 * math.cos(i * 0.7 + t)),
            255
        )

        cv2.circle(img, (px, py), r, color, -1, lineType=cv2.LINE_AA)

def draw_intro_screen(img):
    h, w, _ = img.shape
    base = create_gradient_background(img.shape)

    t = time.time() - program_start_time
    pulse = 0.5 + 0.5 * math.sin(t * 2.2)

    cx = w // 2
    cy = h // 2 - 10

    panel_w = 560
    panel_h = 260
    x1 = cx - panel_w // 2
    y1 = cy - panel_h // 2
    x2 = x1 + panel_w
    y2 = y1 + panel_h

    glow = np.zeros_like(base)
    for k, alpha in [(90, 0.08), (60, 0.12), (34, 0.18)]:
        cv2.rectangle(glow, (x1 - k, y1 - k), (x2 + k, y2 + k), (110, 70, 180), -1)
        glow = cv2.GaussianBlur(glow, (0, 0), 18)
        base = cv2.addWeighted(base, 1.0, glow, alpha, 0)

    pulse_pad = int(8 + pulse * 6)
    ring = np.zeros_like(base)
    cv2.rectangle(
        ring,
        (x1 - pulse_pad, y1 - pulse_pad),
        (x2 + pulse_pad, y2 + pulse_pad),
        (255, 220, 120),
        2
    )
    ring = cv2.GaussianBlur(ring, (0, 0), 3)
    base = cv2.addWeighted(base, 1.0, ring, 0.22, 0)

    draw_pixel_panel_3d(base, x1, y1, x2, y2, body=(42, 24, 70), border=UI_BORDER)

    inner_glow = np.zeros_like(base)
    cv2.rectangle(inner_glow, (x1 + 16, y1 + 16), (x2 - 16, y2 - 16), (90, 40, 120), -1)
    inner_glow = cv2.GaussianBlur(inner_glow, (0, 0), 20)
    base = cv2.addWeighted(base, 1.0, inner_glow, 0.10, 0)

    draw_intro_particles(base, x1 - 8, y1 - 8, x2 + 8, y2 + 8, t)

    title_y = y1 + 58
    subtitle_y = title_y + 30

    line1_y = subtitle_y + 48
    line2_y = line1_y + 34
    line3_y = line2_y + 34

    cta_y = line3_y + 42
    music_y = y2 + 22

    title = APP_NAME
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.10, 3)

    title_glow = np.zeros_like(base)
    cv2.putText(
        title_glow, title,
        (cx - tw // 2, title_y),
        cv2.FONT_HERSHEY_SIMPLEX, 1.10, (255, 220, 120), 6, lineType=cv2.LINE_AA
    )
    title_glow = cv2.GaussianBlur(title_glow, (0, 0), 7)
    base = cv2.addWeighted(base, 1.0, title_glow, 0.18, 0)

    cv2.putText(
        base, title,
        (cx - tw // 2, title_y),
        cv2.FONT_HERSHEY_SIMPLEX, 1.10, UI_TEXT, 3, lineType=cv2.LINE_AA
    )

    subtitle = "DRAW WITH YOUR HAND"
    (sw, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(
        base, subtitle,
        (cx - sw // 2, subtitle_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, UI_DIM, 1, lineType=cv2.LINE_AA
    )

    lines = [
        ("INDEX   -   DRAW", line1_y, 0.72),
        ("PALM    -   MENU", line2_y, 0.72),
        ("PINKY   -   CLEAR", line3_y, 0.72),
    ]

    for text, yy, scale in lines:
        (lw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.putText(
            base, text,
            (cx - lw // 2, yy),
            cv2.FONT_HERSHEY_SIMPLEX, scale, UI_TEXT, 1, lineType=cv2.LINE_AA
        )

    line_w = 250
    line_y = cta_y - 18
    cv2.line(
        base,
        (cx - line_w // 2, line_y),
        (cx + line_w // 2, line_y),
        UI_PANEL_2,
        1,
        lineType=cv2.LINE_AA
    )

    press_text = "PRESS SPACE TO START"
    press_color = UI_BORDER if pulse > 0.5 else UI_TEXT
    (pw, _), _ = cv2.getTextSize(press_text, cv2.FONT_HERSHEY_SIMPLEX, 0.88, 2)

    press_glow = np.zeros_like(base)
    cv2.putText(
        press_glow, press_text,
        (cx - pw // 2, cta_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.88, (255, 220, 120), 5, lineType=cv2.LINE_AA
    )
    press_glow = cv2.GaussianBlur(press_glow, (0, 0), 6)
    base = cv2.addWeighted(base, 1.0, press_glow, 0.16 + pulse * 0.06, 0)

    cv2.putText(
        base, press_text,
        (cx - pw // 2, cta_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.88, press_color, 2, lineType=cv2.LINE_AA
    )

    music_text = "PRESS M TO TOGGLE MUSIC"
    (mw, _), _ = cv2.getTextSize(music_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.putText(
        base, music_text,
        (cx - mw // 2, music_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.40, UI_DIM, 1, lineType=cv2.LINE_AA
    )

    return base

# ======================================
# 11. REPLAY
# ======================================
def finalize_current_stroke():
    global current_stroke, stroke_history
    if len(current_stroke) > 1:
        stroke_history.append(current_stroke.copy())
    current_stroke.clear()

def start_replay(canvas_shape):
    global is_replaying, replay_strokes, replay_stroke_index, replay_point_index, replay_canvas
    if not stroke_history:
        show_action("NO STROKES TO REPLAY", duration=25)
        return

    is_replaying = True
    replay_strokes = [stroke.copy() for stroke in stroke_history]
    replay_stroke_index = 0
    replay_point_index = 1
    replay_canvas = np.zeros(canvas_shape, dtype=np.uint8)
    show_action("REPLAY START", duration=20)

def update_replay(frame_w, frame_h):
    global is_replaying, replay_stroke_index, replay_point_index, replay_canvas

    if not is_replaying or replay_canvas is None:
        return

    if replay_stroke_index >= len(replay_strokes):
        is_replaying = False
        show_action("REPLAY DONE", duration=20)
        return

    stroke = replay_strokes[replay_stroke_index]
    step_count = 3

    for _ in range(step_count):
        if replay_point_index < len(stroke):
            p1 = stroke[replay_point_index - 1]
            p2 = stroke[replay_point_index]
            speed = math.dist(p1, p2)
            draw_brush_line(replay_canvas, p1, p2, speed, brush_base_thickness, frame_w, frame_h)
            replay_point_index += 1
        else:
            replay_stroke_index += 1
            replay_point_index = 1
            break

# ======================================
# 12. MAIN LOOP
# ======================================
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı.")
    raise SystemExit

music_enabled = True
init_audio()

print("Sistem başlatıldı.")
print("SPACE=start | palm=menu | index=draw | hold on button=select | m=music | n=mirror | k=kaleido | q=quit")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Kameradan görüntü alınamadı.")
        break

    img = cv2.flip(img, 1)
    h, w, c = img.shape

    if canvas is None:
        canvas = clear_canvas((h, w, 3))

    if show_intro:
        intro_frame = draw_intro_screen(img)
        cv2.imshow(APP_NAME, intro_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            show_intro = False
            intro_transition_active = True
            intro_transition_start = time.time()
        elif key == ord("q"):
            break
        elif key == ord("m"):
            toggle_music()
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    base_frame = get_base_frame(img)
    overlay_frame = base_frame.copy()

    fingertip_for_menu = None
    menu_selected_button = None
    pointer_pos = None

    if not is_replaying and results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                overlay_frame,
                hand_lms,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(150, 150, 150), thickness=1, circle_radius=1),
                mp_draw.DrawingSpec(color=(90, 90, 90), thickness=1)
            )

            index_up = is_finger_up(hand_lms, 8, 6)
            middle_up = is_finger_up(hand_lms, 12, 10)
            ring_up = is_finger_up(hand_lms, 16, 14)
            pinky_up = is_finger_up(hand_lms, 20, 18)
            palm_open = is_open_palm(hand_lms)
            fist_now = is_fist(hand_lms)

            index_tip = hand_lms.landmark[8]
            raw_x, raw_y = int(index_tip.x * w), int(index_tip.y * h)

            if smooth_x is None or smooth_y is None:
                smooth_x, smooth_y = raw_x, raw_y
            else:
                smooth_x = int((1 - SMOOTHING) * smooth_x + SMOOTHING * raw_x)
                smooth_y = int((1 - SMOOTHING) * smooth_y + SMOOTHING * raw_y)

            cx, cy = smooth_x, smooth_y
            pointer_pos = (cx, cy)
            fingertip_for_menu = (cx, cy)

            if palm_open:
                open_palm_counter += 1
            else:
                open_palm_counter = 0
                open_palm_latch = False

            if open_palm_counter >= OPEN_PALM_HOLD_FRAMES and not open_palm_latch:
                menu_open = not menu_open
                open_palm_latch = True
                points.clear()

                if current_stroke:
                    finalize_current_stroke()

                smooth_x, smooth_y = None, None
                hover_selected_button = None
                hover_start_time = None
                show_action(f"MENU {'OPEN' if menu_open else 'CLOSED'}", duration=18)
                break

            if menu_open:
                if fingertip_for_menu is not None:
                    menu_selected_button = get_hovered_button(fingertip_for_menu[0], fingertip_for_menu[1])

                current_time = time.time()

                if menu_selected_button is None:
                    hover_selected_button = None
                    hover_start_time = None
                else:
                    if hover_selected_button != menu_selected_button:
                        hover_selected_button = menu_selected_button
                        hover_start_time = current_time
                    else:
                        if hover_start_time is not None:
                            elapsed = current_time - hover_start_time
                            if elapsed >= HOVER_SELECT_SECONDS:
                                execute_menu_action(menu_selected_button, w, h)

                points.clear()
                if current_stroke:
                    finalize_current_stroke()
                smooth_x, smooth_y = None, None
                break

            only_index = index_up and (not middle_up) and (not ring_up) and (not pinky_up)

            if only_index:
                fist_was_active = False
                points.append((cx, cy))
                current_stroke.append((cx, cy))

                if len(points) > 1:
                    p1 = points[-2]
                    p2 = points[-1]
                    speed = math.dist(p1, p2)
                    draw_brush_line(canvas, p1, p2, speed, brush_base_thickness, w, h)

                add_draw_sparks(overlay_frame, cx, cy, get_current_palette())
                particles.append(Particle(cx, cy, mode=effect_mode, burst=False))

            elif fist_now:
                if not fist_was_active:
                    fist_was_active = True
                    finalize_current_stroke()

                    for _ in range(35):
                        particles.append(Particle(cx, cy, mode=effect_mode, burst=True))

                    show_action("BURST", duration=16)
                    points.clear()
                    smooth_x, smooth_y = None, None

            elif pinky_up and (not index_up) and (not middle_up) and (not ring_up):
                fist_was_active = False
                canvas = clear_canvas((h, w, 3))
                points.clear()
                current_stroke = []
                stroke_history = []
                smooth_x, smooth_y = None, None
                replay_canvas = None
                is_replaying = False
                show_action("CANVAS CLEARED", duration=20)

            else:
                if current_stroke:
                    finalize_current_stroke()
                points.clear()
                fist_was_active = False

    else:
        if current_stroke:
            finalize_current_stroke()
        fist_was_active = False
        smooth_x, smooth_y = None, None
        open_palm_latch = False
        open_palm_counter = 0
        hover_selected_button = None
        hover_start_time = None
        points.clear()

    if is_replaying:
        update_replay(w, h)

    for p in particles[:]:
        p.move()
        if p.life <= 0:
            particles.remove(p)
        else:
            alpha_scale = max(0.2, p.life / 255.0)
            radius = max(1, int(p.radius * alpha_scale))
            cv2.circle(
                overlay_frame,
                (int(p.x), int(p.y)),
                radius,
                p.color,
                -1,
                lineType=cv2.LINE_AA
            )

    if pointer_pos is not None:
        draw_pointer_glow(overlay_frame, pointer_pos[0], pointer_pos[1], get_current_palette())

    clean_frame = cv2.addWeighted(overlay_frame, 1.0, canvas, 0.98, 0)

    if is_replaying and replay_canvas is not None:
        clean_frame = cv2.addWeighted(clean_frame, 1.0, replay_canvas, 1.0, 0)

    latest_clean_frame = clean_frame.copy()

    final_frame = clean_frame.copy()
    draw_hud(final_frame, effect_mode, brush_base_thickness)
    draw_recording_badge(final_frame)
    draw_status_text(final_frame)

    if menu_open:
        draw_gesture_menu(final_frame, fingertip_for_menu, hover_selected_button)

    if intro_transition_active and intro_transition_start is not None:
        final_frame, done = apply_intro_transition(
            final_frame,
            intro_transition_start,
            INTRO_TRANSITION_DURATION
        )
        if done:
            intro_transition_active = False
            intro_transition_start = None

    if is_recording and video_writer is not None:
        video_writer.write(final_frame)

    cv2.imshow(APP_NAME, final_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("1"):
        effect_mode = "spark"
        show_action("EFFECT SPARK", duration=20)

    elif key == ord("2"):
        effect_mode = "magic"
        show_action("EFFECT MAGIC", duration=20)

    elif key == ord("3"):
        effect_mode = "fire"
        show_action("EFFECT FIRE", duration=20)

    elif key == ord("t"):
        cycle_brush_style()

    elif key == ord("r"):
        cycle_color()

    elif key == ord("b"):
        cycle_background()

    elif key == ord("["):
        brush_base_thickness = max(2, brush_base_thickness - 1)
        show_action(f"SIZE {brush_base_thickness}", duration=20)

    elif key == ord("]"):
        brush_base_thickness = min(12, brush_base_thickness + 1)
        show_action(f"SIZE {brush_base_thickness}", duration=20)

    elif key == ord("p"):
        start_replay((h, w, 3))

    elif key == ord("c"):
        canvas = clear_canvas((h, w, 3))
        points.clear()
        current_stroke = []
        stroke_history = []
        replay_canvas = None
        is_replaying = False
        show_action("ALL CLEARED", duration=20)

    elif key == ord("s"):
        save_screenshot(final_frame)

    elif key == ord("v"):
        toggle_video_recording(w, h)

    elif key == ord("e"):
        if latest_clean_frame is not None:
            export_session(latest_clean_frame, canvas)

    elif key == ord("n"):
        toggle_mirror()

    elif key == ord("k"):
        toggle_kaleido()

    elif key == ord("m"):
        toggle_music()

    elif key == ord("q"):
        break

if video_writer is not None:
    video_writer.release()

cap.release()
cv2.destroyAllWindows()

if audio_ready:
    pygame.mixer.music.stop()
    pygame.mixer.quit()