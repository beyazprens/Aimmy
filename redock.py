"""
redock.py — C++ Redock.exe'nin Python versiyonu
Driver: \\.\volmgra (IO_SEND_MOUSE_EVENT = 0x1FE3BD28)
Mimari: Ayrı thread'de capture + inference, ana thread'de mouse hareketi
"""

import ctypes
import ctypes.wintypes
import threading
import time
import math
import os
import configparser
import sys
import queue
from datetime import datetime

import numpy as np
import onnxruntime as ort
import mss
import cv2
import tkinter as tk

# ──────────────────────────────────────────────────────────────
# SABITLER VE VARSAYILAN AYARLAR
# ──────────────────────────────────────────────────────────────
MODEL_PATH          = "best_fp16.onnx"
INPUT_SIZE          = 320
CONFIG_FILE         = "settings.cfg"

FOV                 = 150
CONFIDENCE          = 0.65
AIM_SMOOTH          = 5.0
AIM_SENS            = 1.0
DEADZONE            = 3
HEAD_OFFSET         = 0.06
PREDICTION_FACTOR   = 0.45

RECOIL_ENABLED      = True
RECOIL_STRENGTH     = 1.0
TRIGGERBOT_ENABLED  = False
TRIGGER_AUTO_MODE   = False
SCREENSHOTS_ENABLED = True
FRAME_LIMIT         = 144
FOV_OVERLAY_VISIBLE = True

MAX_MOVE_SPEED      = 45.0
STICKY_DISTANCE     = 45.0
TARGET_LOCK_BONUS   = 60.0
MAX_LOST_FRAMES     = 5

# Triggerbot zamanlama sabitleri (C++'dan alınan değerler)
TRIGGER_RADIUS         = 25        # piksel cinsinden tetikleyici yarıçapı
TRIGGER_COOLDOWN       = 0.120     # single mode: ateşler arası minimum bekleme (saniye)
TRIGGER_PRESS_DURATION = 0.025     # single mode: LMB basılı tutma süresi (saniye)

# Recoil baz değeri (C++ ölçümlere dayalı sabit)
RECOIL_BASE = 6.2

DEVICE_NAME         = r"\\.\volmgra"
IO_SEND_MOUSE_EVENT = 0x1FE3BD28

# Numpad sanal tuş kodları
VK_NUMPAD0  = 0x60
VK_NUMPAD1  = 0x61
VK_NUMPAD2  = 0x62
VK_NUMPAD3  = 0x63
VK_NUMPAD4  = 0x64
VK_NUMPAD5  = 0x65
VK_NUMPAD6  = 0x66
VK_NUMPAD7  = 0x67
VK_NUMPAD8  = 0x68
VK_NUMPAD9  = 0x69
VK_MULTIPLY = 0x6A  # NUMPAD *
VK_ADD      = 0x6B  # NUMPAD +
VK_SUBTRACT = 0x6D  # NUMPAD -
VK_DIVIDE   = 0x6F  # NUMPAD /
VK_LBUTTON  = 0x01
VK_RBUTTON  = 0x02


# ──────────────────────────────────────────────────────────────
# DRIVER (volmgra)
# ──────────────────────────────────────────────────────────────
class NF_MOUSE_REQUEST(ctypes.Structure):
    _fields_ = [
        ("x",           ctypes.c_int),
        ("y",           ctypes.c_int),
        ("ButtonFlags", ctypes.c_short),
    ]


class Driver:
    """volmgra sürücüsüne IOCTL ile mouse hareketi gönderir."""

    GENERIC_READ    = 0x80000000
    GENERIC_WRITE   = 0x40000000
    OPEN_EXISTING   = 3
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE= 0x00000002

    def __init__(self):
        self.handle = None
        self._open()

    def _open(self):
        self.handle = ctypes.windll.kernel32.CreateFileW(
            DEVICE_NAME,
            self.GENERIC_READ | self.GENERIC_WRITE,
            self.FILE_SHARE_READ | self.FILE_SHARE_WRITE,
            None,
            self.OPEN_EXISTING,
            0,
            None,
        )
        if self.handle == ctypes.wintypes.HANDLE(-1).value:
            error_msg = ctypes.FormatError(ctypes.GetLastError())
            print(f"[UYARI] Driver açılamadı: {error_msg}")
            self.handle = None

    def move(self, x: int, y: int, button_flags: int = 0):
        if self.handle is None:
            return
        req = NF_MOUSE_REQUEST(x=x, y=y, ButtonFlags=button_flags)
        bytes_returned = ctypes.c_ulong(0)
        ctypes.windll.kernel32.DeviceIoControl(
            self.handle,
            IO_SEND_MOUSE_EVENT,
            ctypes.byref(req),
            ctypes.sizeof(req),
            None,
            0,
            ctypes.byref(bytes_returned),
            None,
        )

    def click(self, button_flags: int):
        """Yalnızca buton flag'i ile IOCTL gönder (x=0, y=0)."""
        self.move(0, 0, button_flags)

    def close(self):
        if self.handle:
            ctypes.windll.kernel32.CloseHandle(self.handle)
            self.handle = None


# ──────────────────────────────────────────────────────────────
# KALMAN FİLTRE
# ──────────────────────────────────────────────────────────────
class KalmanFilter2D:
    """2D Kalman filtresi (x, y koordinat tahmini)."""

    def __init__(self, process_noise=1e-3, measurement_noise=1e-1, error=1.0):
        # Durum: [x, y, vx, vy]
        self.x = np.zeros((4, 1), dtype=np.float64)
        # Geçiş matrisi
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        # Ölçüm matrisi
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)
        self.Q = np.eye(4, dtype=np.float64) * process_noise
        self.R = np.eye(2, dtype=np.float64) * measurement_noise
        self.P = np.eye(4, dtype=np.float64) * error

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].flatten()

    def update(self, z: np.ndarray):
        """z: [x, y] ölçüm."""
        z = z.reshape((2, 1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2].flatten()

    def reset(self):
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64)


# ──────────────────────────────────────────────────────────────
# BEZİER YUMUŞATMASı
# ──────────────────────────────────────────────────────────────
def bezier_point(p0, p1, p2, t: float):
    """İkinci dereceden Bezier eğrisi üzerindeki nokta."""
    return (
        (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0],
        (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1],
    )


def bezier_move(driver: Driver, dx: float, dy: float, steps: int = 6):
    """Hareketi Bezier eğrisiyle birden fazla adıma bölerek uygular."""
    if dx == 0 and dy == 0:
        return
    p0 = (0.0, 0.0)
    p2 = (dx, dy)
    # Orta kontrol noktası hafif sapma
    p1 = (dx * 0.5 + dy * 0.15, dy * 0.5 - dx * 0.15)
    prev = p0
    for i in range(1, steps + 1):
        t = i / steps
        curr = bezier_point(p0, p1, p2, t)
        step_x = int(round(curr[0] - prev[0]))
        step_y = int(round(curr[1] - prev[1]))
        if step_x != 0 or step_y != 0:
            driver.move(step_x, step_y)
        prev = curr


# ──────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ──────────────────────────────────────────────────────────────
def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def key_pressed(vk: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def key_just_pressed(vk: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x0001)


# ──────────────────────────────────────────────────────────────
# AYAR YÖNETİMİ (settings.cfg)
# ──────────────────────────────────────────────────────────────
_cfg_lock = threading.Lock()


def load_settings():
    global FOV, CONFIDENCE, AIM_SMOOTH, AIM_SENS, DEADZONE, HEAD_OFFSET
    global PREDICTION_FACTOR, RECOIL_ENABLED, RECOIL_STRENGTH
    global TRIGGERBOT_ENABLED, TRIGGER_AUTO_MODE, SCREENSHOTS_ENABLED
    global FRAME_LIMIT, FOV_OVERLAY_VISIBLE, MAX_MOVE_SPEED
    global STICKY_DISTANCE, TARGET_LOCK_BONUS, MAX_LOST_FRAMES

    cfg = configparser.ConfigParser()
    if not os.path.exists(CONFIG_FILE):
        save_settings()
        return

    cfg.read(CONFIG_FILE)
    s = cfg["Settings"] if "Settings" in cfg else {}

    FOV                 = float(s.get("FOV",                 FOV))
    CONFIDENCE          = float(s.get("CONFIDENCE",          CONFIDENCE))
    AIM_SMOOTH          = float(s.get("AIM_SMOOTH",          AIM_SMOOTH))
    AIM_SENS            = float(s.get("AIM_SENS",            AIM_SENS))
    DEADZONE            = int(  s.get("DEADZONE",            DEADZONE))
    HEAD_OFFSET         = float(s.get("HEAD_OFFSET",         HEAD_OFFSET))
    PREDICTION_FACTOR   = float(s.get("PREDICTION_FACTOR",   PREDICTION_FACTOR))
    RECOIL_ENABLED      = s.get("RECOIL_ENABLED",      str(RECOIL_ENABLED)).lower() == "true"
    RECOIL_STRENGTH     = float(s.get("RECOIL_STRENGTH",     RECOIL_STRENGTH))
    TRIGGERBOT_ENABLED  = s.get("TRIGGERBOT_ENABLED",  str(TRIGGERBOT_ENABLED)).lower() == "true"
    TRIGGER_AUTO_MODE   = s.get("TRIGGER_AUTO_MODE",   str(TRIGGER_AUTO_MODE)).lower() == "true"
    SCREENSHOTS_ENABLED = s.get("SCREENSHOTS_ENABLED", str(SCREENSHOTS_ENABLED)).lower() == "true"
    FRAME_LIMIT         = int(  s.get("FRAME_LIMIT",         FRAME_LIMIT))
    FOV_OVERLAY_VISIBLE = s.get("FOV_OVERLAY_VISIBLE",  str(FOV_OVERLAY_VISIBLE)).lower() == "true"
    MAX_MOVE_SPEED      = float(s.get("MAX_MOVE_SPEED",      MAX_MOVE_SPEED))
    STICKY_DISTANCE     = float(s.get("STICKY_DISTANCE",     STICKY_DISTANCE))
    TARGET_LOCK_BONUS   = float(s.get("TARGET_LOCK_BONUS",   TARGET_LOCK_BONUS))
    MAX_LOST_FRAMES     = int(  s.get("MAX_LOST_FRAMES",     MAX_LOST_FRAMES))


def save_settings():
    with _cfg_lock:
        cfg = configparser.ConfigParser()
        cfg["Settings"] = {
            "FOV":                 str(FOV),
            "CONFIDENCE":          str(CONFIDENCE),
            "AIM_SMOOTH":          str(AIM_SMOOTH),
            "AIM_SENS":            str(AIM_SENS),
            "DEADZONE":            str(DEADZONE),
            "HEAD_OFFSET":         str(HEAD_OFFSET),
            "PREDICTION_FACTOR":   str(PREDICTION_FACTOR),
            "RECOIL_ENABLED":      str(RECOIL_ENABLED),
            "RECOIL_STRENGTH":     str(RECOIL_STRENGTH),
            "TRIGGERBOT_ENABLED":  str(TRIGGERBOT_ENABLED),
            "TRIGGER_AUTO_MODE":   str(TRIGGER_AUTO_MODE),
            "SCREENSHOTS_ENABLED": str(SCREENSHOTS_ENABLED),
            "FRAME_LIMIT":         str(FRAME_LIMIT),
            "FOV_OVERLAY_VISIBLE": str(FOV_OVERLAY_VISIBLE),
            "MAX_MOVE_SPEED":      str(MAX_MOVE_SPEED),
            "STICKY_DISTANCE":     str(STICKY_DISTANCE),
            "TARGET_LOCK_BONUS":   str(TARGET_LOCK_BONUS),
            "MAX_LOST_FRAMES":     str(MAX_LOST_FRAMES),
        }
        with open(CONFIG_FILE, "w") as f:
            cfg.write(f)


# ──────────────────────────────────────────────────────────────
# KONSOL MENÜ
# ──────────────────────────────────────────────────────────────
def print_menu():
    os.system("cls" if os.name == "nt" else "clear")
    on_off = lambda v: "ON " if v else "OFF"
    print("=====================================================")
    print("                INTERNAL SYSTEM LOADED               ")
    print("=====================================================")
    print(f" [NUM /]   Recoil Assist:      {on_off(RECOIL_ENABLED)}")
    print(f" [NUM 8]   Triggerbot:         {on_off(TRIGGERBOT_ENABLED)}")
    print(f" [NUM 9]   Trigger Mode:       {'AUTO  ' if TRIGGER_AUTO_MODE else 'SINGLE'}")
    print(f" [NUM 7]   FOV Overlay:        {on_off(FOV_OVERLAY_VISIBLE)}")
    print(f" [NUM *]   Screenshots:        {on_off(SCREENSHOTS_ENABLED)}")
    print(f" [NUM 5]   Restart/Reload")
    print(f" [NUM 0]   Exit")
    print("-----------------------------------------------------")
    print(f" [NUM 1] +/-  Smoothing:       {AIM_SMOOTH:.2f}")
    print(f" [NUM 2] +/-  Sensitivity:     {AIM_SENS:.2f}")
    print(f" [NUM 3] +/-  FOV Radius:      {int(FOV)}")
    print(f" [NUM 4] +/-  Frame Limit:     {FRAME_LIMIT}")
    print(f" [NUM 6] +/-  Confidence:      {CONFIDENCE:.2f}")
    print(f"        +/-   Recoil Strength: {RECOIL_STRENGTH:.2f}")
    print("=====================================================")


# ──────────────────────────────────────────────────────────────
# FOV OVERLAY (Tkinter)
# ──────────────────────────────────────────────────────────────
class FovOverlay:
    """Ekranın ortasına yarı saydam FOV dairesi çizen Tkinter penceresi."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", "black")
        self.root.configure(bg="black")

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{sw}x{sh}+0+0")

        self.canvas = tk.Canvas(
            self.root, width=sw, height=sh,
            bg="black", highlightthickness=0
        )
        self.canvas.pack()

        self.sw = sw
        self.sh = sh
        self._circle = None
        self._draw()

    def _draw(self):
        self.canvas.delete("all")
        cx, cy = self.sw // 2, self.sh // 2
        r = int(FOV)
        self._circle = self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline="#00FF00", width=2
        )

    def update(self, visible: bool):
        if visible:
            self.root.deiconify()
            self._draw()
        else:
            self.root.withdraw()
        self.root.update()

    def destroy(self):
        self.root.destroy()


# ──────────────────────────────────────────────────────────────
# EKRAN YAKALAMA + INFERENCE THREAD
# ──────────────────────────────────────────────────────────────
class Detection:
    """Tek bir tespit sonucu."""
    __slots__ = ("x", "y", "w", "h", "conf")

    def __init__(self, x, y, w, h, conf):
        self.x    = x
        self.y    = y
        self.w    = w
        self.h    = h
        self.conf = conf


class CaptureThread(threading.Thread):
    """Ekranı FOV bölgesinde yakalar, ONNX ile inference yapar, kuyruğa koyar."""

    def __init__(self, result_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True, name="CaptureThread")
        self.result_queue = result_queue
        self.stop_event   = stop_event
        self._session     = None
        self._sct         = None

    def _load_model(self):
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=providers)
        except Exception as e:
            # DML sağlayıcısı mevcut değilse CPU'ya geri dön
            print(f"[UYARI] DML sağlayıcısı kullanılamıyor: {e}")
            session = ort.InferenceSession(MODEL_PATH, sess_options=opts,
                                           providers=["CPUExecutionProvider"])
        self._session = session
        self._input_name = session.get_inputs()[0].name

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Önce float32 ile normalize et, sonra float16'ya çevir (precision loss önlenir)
        img = (img.astype(np.float32) / 255.0).astype(np.float16)
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, 0)

    def _postprocess(self, output, orig_w, orig_h):
        """YOLOv8 çıktısını Detection listesine çevirir."""
        detections = []
        preds = output[0]  # (1, 5, N) veya (1, N, 5)

        # YOLOv8 çıktı şeklini normalize et
        if preds.ndim == 3 and preds.shape[0] == 1:
            preds = preds[0]  # (5, N) veya (N, 5)

        if preds.shape[0] == 5:
            preds = preds.T  # (N, 5)

        sx = orig_w / INPUT_SIZE
        sy = orig_h / INPUT_SIZE

        for row in preds:
            cx, cy, bw, bh, conf = row
            if conf < CONFIDENCE:
                continue
            x = (cx - bw / 2) * sx
            y = (cy - bh / 2) * sy
            w = bw * sx
            h = bh * sy
            detections.append(Detection(x, y, w, h, float(conf)))

        return detections

    def run(self):
        self._load_model()
        self._sct = mss.mss()

        while not self.stop_event.is_set():
            frame_start = time.perf_counter()

            # Ekran çözünürlüğü al
            mon_info = self._sct.monitors[1]
            sw = mon_info["width"]
            sh = mon_info["height"]
            cx = sw // 2
            cy = sh // 2
            fov_r = int(FOV)

            # FOV bölgesi
            left   = max(0, cx - fov_r)
            top    = max(0, cy - fov_r)
            right  = min(sw, cx + fov_r)
            bottom = min(sh, cy + fov_r)
            region_w = right - left
            region_h = bottom - top

            monitor = {"left": left, "top": top, "width": region_w, "height": region_h}
            raw = self._sct.grab(monitor)
            img_bgr = np.frombuffer(raw.raw, dtype=np.uint8).reshape(raw.height, raw.width, 4)
            img_bgr = img_bgr[:, :, :3]  # BGRA → BGR

            inp = self._preprocess(img_bgr)
            out = self._session.run(None, {self._input_name: inp})
            dets = self._postprocess(out, region_w, region_h)

            # Kuyruğa koy (taşınca eski atılır)
            payload = {
                "detections": dets,
                "region":     (left, top, region_w, region_h),
                "screen":     (sw, sh),
                "timestamp":  time.perf_counter(),
            }
            if not self.result_queue.full():
                self.result_queue.put_nowait(payload)
            else:
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
                self.result_queue.put_nowait(payload)

            # Frame limiti
            elapsed = time.perf_counter() - frame_start
            frame_time = 1.0 / FRAME_LIMIT
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)


# ──────────────────────────────────────────────────────────────
# SCREENSHOT YÖNETİCİSİ
# ──────────────────────────────────────────────────────────────
class ScreenshotManager:
    def __init__(self):
        self._last_shot = 0.0
        os.makedirs("screenshots", exist_ok=True)

    def try_capture(self, sct):
        if not SCREENSHOTS_ENABLED:
            return
        if not (key_pressed(VK_RBUTTON) and key_pressed(VK_LBUTTON)):
            return
        now = time.time()
        if now - self._last_shot < 1.0:
            return
        self._last_shot = now
        mon = sct.monitors[1]
        raw = sct.grab(mon)
        img = np.frombuffer(raw.raw, dtype=np.uint8).reshape(raw.height, raw.width, 4)
        img = img[:, :, :3]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join("screenshots", f"shot_{timestamp}.png")
        cv2.imwrite(path, img)


# ──────────────────────────────────────────────────────────────
# ANA AIM LOOP
# ──────────────────────────────────────────────────────────────
def main():
    global FOV, CONFIDENCE, AIM_SMOOTH, AIM_SENS, RECOIL_ENABLED, RECOIL_STRENGTH
    global TRIGGERBOT_ENABLED, TRIGGER_AUTO_MODE, SCREENSHOTS_ENABLED
    global FRAME_LIMIT, FOV_OVERLAY_VISIBLE

    load_settings()
    print_menu()

    driver = Driver()

    # Kalman filtresi
    kf = KalmanFilter2D()

    # Smoothing durumu
    cur_dx = 0.0
    cur_dy = 0.0

    # Target lock durumu
    locked_x       = None
    locked_y       = None
    frames_lost    = 0

    # Önceki frame bilgisi
    last_best_x    = None

    # Triggerbot durumu
    trigger_pressed   = False
    trigger_last_shot = 0.0

    # Screenshot yöneticisi
    shot_mgr = ScreenshotManager()

    # Capture thread
    result_queue = queue.Queue(maxsize=2)
    stop_event   = threading.Event()
    capture = CaptureThread(result_queue, stop_event)
    capture.start()

    # FOV overlay (Tkinter — main thread'de)
    overlay = FovOverlay()
    if not FOV_OVERLAY_VISIBLE:
        overlay.update(False)

    # Numpad state takibi
    numpad_held_key = None
    numpad_last_adj = 0.0

    mss_sct = mss.mss()
    menu_dirty = False

    print("[INFO] Redock başlatıldı. Çıkmak için NUMPAD 0.")

    try:
        while True:
            loop_start = time.perf_counter()

            # ── NUMPAD KONTROLLER ──────────────────────────────
            changed = False

            # Toggle tuşları (tek basış)
            if key_just_pressed(VK_DIVIDE):      # NUM /
                RECOIL_ENABLED = not RECOIL_ENABLED
                changed = True

            if key_just_pressed(VK_NUMPAD8):
                TRIGGERBOT_ENABLED = not TRIGGERBOT_ENABLED
                changed = True

            if key_just_pressed(VK_NUMPAD9):
                TRIGGER_AUTO_MODE = not TRIGGER_AUTO_MODE
                changed = True

            if key_just_pressed(VK_NUMPAD7):
                FOV_OVERLAY_VISIBLE = not FOV_OVERLAY_VISIBLE
                overlay.update(FOV_OVERLAY_VISIBLE)
                changed = True

            if key_just_pressed(VK_MULTIPLY):    # NUM *
                SCREENSHOTS_ENABLED = not SCREENSHOTS_ENABLED
                changed = True

            if key_just_pressed(VK_NUMPAD5):
                # Yeniden yükle
                load_settings()
                kf.reset()
                cur_dx = cur_dy = 0.0
                locked_x = locked_y = None
                last_best_x = None
                changed = True

            if key_just_pressed(VK_NUMPAD0):
                print("[INFO] Çıkılıyor...")
                break

            # Ayar tuşları (basılı + +/-)
            plus  = key_pressed(VK_ADD)
            minus = key_pressed(VK_SUBTRACT)
            direction = 0
            if plus:
                direction = 1
            elif minus:
                direction = -1

            if direction != 0:
                now_t = time.perf_counter()
                delay = 0.15 if now_t - numpad_last_adj > 0.5 else 0.08

                if key_pressed(VK_NUMPAD1):
                    if now_t - numpad_last_adj > delay:
                        AIM_SMOOTH = clamp(AIM_SMOOTH + direction * 0.1, 0.1, 20.0)
                        numpad_last_adj = now_t; changed = True

                elif key_pressed(VK_NUMPAD2):
                    if now_t - numpad_last_adj > delay:
                        AIM_SENS = clamp(AIM_SENS + direction * 0.05, 0.1, 5.0)
                        numpad_last_adj = now_t; changed = True

                elif key_pressed(VK_NUMPAD3):
                    if now_t - numpad_last_adj > delay:
                        FOV = clamp(FOV + direction * 5, 30, 400)
                        overlay.update(FOV_OVERLAY_VISIBLE)
                        numpad_last_adj = now_t; changed = True

                elif key_pressed(VK_NUMPAD4):
                    if now_t - numpad_last_adj > delay:
                        FRAME_LIMIT = int(clamp(FRAME_LIMIT + direction * 10, 10, 360))
                        numpad_last_adj = now_t; changed = True

                elif key_pressed(VK_NUMPAD6):
                    if now_t - numpad_last_adj > delay:
                        CONFIDENCE = clamp(CONFIDENCE + direction * 0.01, 0.1, 0.99)
                        numpad_last_adj = now_t; changed = True

                else:
                    # Tek başına +/- → recoil strength
                    if now_t - numpad_last_adj > delay:
                        RECOIL_STRENGTH = clamp(RECOIL_STRENGTH + direction * 0.1, 0.0, 5.0)
                        numpad_last_adj = now_t; changed = True

            if changed:
                save_settings()
                print_menu()

            # ── DETECTION SONUCU AL ──────────────────────────
            payload = None
            try:
                payload = result_queue.get_nowait()
            except queue.Empty:
                pass

            # RMB basılı mı? (aim aktif)
            aiming = key_pressed(VK_RBUTTON)

            if payload is None or not aiming:
                # Hedef yok veya aim kapalı — smoothing sıfırla
                cur_dx *= 0.7
                cur_dy *= 0.7
                if not aiming:
                    locked_x = locked_y = None
                    frames_lost = 0
                    last_best_x = None
                    kf.reset()
                overlay.update(FOV_OVERLAY_VISIBLE)
                # Screenshot fırsatı
                shot_mgr.try_capture(mss_sct)
                time.sleep(0.001)
                continue

            dets: list = payload["detections"]
            reg_left, reg_top, reg_w, reg_h = payload["region"]
            sw, sh = payload["screen"]
            cx = sw // 2
            cy = sh // 2

            # ── HEDEF SEÇİMİ (target locking) ────────────────
            best = None
            best_score = -1.0

            fov_r = float(FOV)
            for d in dets:
                # Detection koordinatları bölgeye göre — ekran koordinatlarına çevir
                target_cx = reg_left + d.x + d.w / 2.0
                target_cy = reg_top  + d.y + d.h * (1.0 - HEAD_OFFSET)

                dist = math.hypot(target_cx - cx, target_cy - cy)
                if dist > fov_r:
                    continue

                score = d.conf - dist / fov_r * 0.3

                # Önceki kilit varsa bonus
                if locked_x is not None:
                    lock_dist = math.hypot(target_cx - locked_x, target_cy - locked_y)
                    if lock_dist < STICKY_DISTANCE:
                        score += TARGET_LOCK_BONUS / 1000.0

                if score > best_score:
                    best_score = score
                    best = (target_cx, target_cy, d.w, d.h)

            if best is None:
                frames_lost += 1
                if frames_lost > MAX_LOST_FRAMES:
                    locked_x = locked_y = None
                    last_best_x = None
                    kf.reset()
                overlay.update(FOV_OVERLAY_VISIBLE)
                shot_mgr.try_capture(mss_sct)
                continue

            frames_lost = 0
            best_x, best_y, best_w, best_h = best

            # Kalman güncelle
            kf_pos = kf.update(np.array([best_x, best_y]))
            kf_pred = kf.predict()
            kf_x, kf_y = kf_pred

            # Kilit güncelle
            locked_x, locked_y = kf_x, kf_y

            # ── AIM ALGORİTMASI (C++ mantığı) ────────────────
            aim_dx = kf_x - cx
            aim_dy = kf_y - cy

            # Deadzone kontrolü
            if abs(aim_dx) < DEADZONE and abs(aim_dy) < DEADZONE:
                overlay.update(FOV_OVERLAY_VISIBLE)
                shot_mgr.try_capture(mss_sct)
                continue

            # Aspect ratio dinamik ölçekleme
            # 2.5 böleni: dikey hitbox'ı yatay boyutla karşılaştırılabilir hale getirir
            w = best_w
            h = best_h
            if w > h * 0.8:
                effective_dim = max(w, h / 2.5)
            else:
                effective_dim = h / 2.5

            dist_factor    = clamp(effective_dim / 35.0, 0.45, 1.0)
            dynamic_sens   = AIM_SENS * dist_factor
            dynamic_smooth = clamp(AIM_SMOOTH / dist_factor, 0.1, 0.95)

            # Velocity tahmini (x ekseninde)
            vel_x = 0.0
            if last_best_x is not None:
                vel_x = best_x - last_best_x
            last_best_x = best_x

            target_dx = (aim_dx + vel_x * PREDICTION_FACTOR) * dynamic_sens
            target_dy = aim_dy * dynamic_sens

            # Exponential smoothing (C++ tarzı)
            cur_dx += (target_dx - cur_dx) * (1.0 - dynamic_smooth)
            cur_dy += (target_dy - cur_dy) * (1.0 - dynamic_smooth)

            # Clamp
            final_move_x = clamp(cur_dx, -MAX_MOVE_SPEED, MAX_MOVE_SPEED)
            final_move_y = clamp(cur_dy, -MAX_MOVE_SPEED, MAX_MOVE_SPEED)

            # Recoil kompanzasyonu (RECOIL_BASE: C++ ölçümlere dayalı sabit dikey itme)
            is_shooting_held = key_pressed(VK_LBUTTON)
            is_auto_firing   = TRIGGERBOT_ENABLED and TRIGGER_AUTO_MODE and best is not None
            if RECOIL_ENABLED and (is_shooting_held or is_auto_firing):
                final_move_y += RECOIL_BASE * RECOIL_STRENGTH

            # Sub-pixel düzeltmesi (C++ mantığı)
            if abs(final_move_x) >= 0.5 and round(final_move_x) == 0:
                final_move_x = 1.0 if final_move_x > 0 else -1.0
            if abs(final_move_y) >= 0.5 and round(final_move_y) == 0:
                final_move_y = 1.0 if final_move_y > 0 else -1.0

            move_x = int(round(final_move_x))
            move_y = int(round(final_move_y))

            # Bezier hareketi uygula (Kalman → exponential → Bezier)
            if move_x != 0 or move_y != 0:
                bezier_move(driver, final_move_x, final_move_y, steps=4)

            # ── TRİGGERBOT ───────────────────────────────────
            if TRIGGERBOT_ENABLED and best is not None:
                in_trigger = (abs(aim_dx) < TRIGGER_RADIUS and abs(aim_dy) < TRIGGER_RADIUS)

                if TRIGGER_AUTO_MODE:
                    # LMB bas/bırak hedef içindeyken
                    if in_trigger and not trigger_pressed:
                        driver.click(0x0001)   # LMB down (flag)
                        trigger_pressed = True
                    elif not in_trigger and trigger_pressed:
                        driver.click(0x0002)   # LMB up (flag)
                        trigger_pressed = False
                else:
                    # Single shot modu: cooldown & basış süresi sabitlere göre
                    now_t = time.perf_counter()
                    if in_trigger and not trigger_pressed and (now_t - trigger_last_shot) > TRIGGER_COOLDOWN:
                        driver.click(0x0001)
                        trigger_pressed   = True
                        trigger_last_shot = now_t
                    if trigger_pressed and (time.perf_counter() - trigger_last_shot) > TRIGGER_PRESS_DURATION:
                        driver.click(0x0002)
                        trigger_pressed = False

            # ── FOV OVERLAY & SCREENSHOT ──────────────────────
            overlay.update(FOV_OVERLAY_VISIBLE)
            shot_mgr.try_capture(mss_sct)

    finally:
        stop_event.set()
        capture.join(timeout=3)
        driver.close()
        try:
            overlay.destroy()
        except Exception as e:
            print(f"[UYARI] Overlay kapatılırken hata: {e}")
        save_settings()
        print("[INFO] Redock kapatıldı.")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
