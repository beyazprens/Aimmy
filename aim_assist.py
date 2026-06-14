"""
Python Aim Assist - Pro Edition
Kalman filter, Bezier smoothing, multi-thread, head targeting.
Mouse2 basili tutunca aktif.
"""

import ctypes
import ctypes.wintypes as wintypes
import time
import math
import os
import threading
import numpy as np
import cv2
import mss
import onnxruntime as ort
from collections import deque

# ╔═══════════════════════════════════════════════════════╗
# ║                    AYARLAR                            ║
# ╠═══════════════════════════════════════════════════════╣
MODEL_PATH = os.path.abspath("1.5kR6.onnx")
INPUT_SIZE = 640              # Model input boyutu

# -- Aim --
FOV = 320                     # Tarama alani (piksel)
CONFIDENCE = 0.40             # Minimum confidence
AIM_SMOOTH = 5.0              # Smoothness (1=anlik, 10=cok yavas)
DEADZONE = 3                  # Piksel - bu kadar yakinsa hareket etme
HEAD_OFFSET = 0.30            # Bbox'in ustunden %30'a aim (kafa)
PREDICTION_FACTOR = 0.15      # Hedef hiz tahmini carpani

# -- Bezier --
BEZIER_STEPS = 3              # Kac adimda hedefe ulassin (2-5 arasi)
BEZIER_JITTER = 0.8           # Rastgelelik (0=yok, 2=cok - insan gibi)

# -- FOV Overlay --
SHOW_FOV = True               # True = yuvarlak FOV gozukur, False = gizli
FOV_COLOR = (0, 255, 0)      # Renk (R, G, B) - yesil
FOV_THICKNESS = 2             # Cizgi kalinligi (piksel)
FOV_OPACITY = 180             # Saydamlik (0-255, 255=tam gorunur)

# -- Performance --
MAX_FPS = 120                 # Maksimum dongu FPS limiti
# ╚═══════════════════════════════════════════════════════╝

# === Driver ===
DEVICE_NAME = r"\\.\volmgra"
IO_SEND_MOUSE_EVENT = 0x1FE3BD28
VK_RBUTTON = 0x02


class NF_MOUSE_REQUEST(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int), ("ButtonFlags", ctypes.c_short)]


# ═══════════════════════════════════════════════════════
#                    DRIVER CLASS
# ═══════════════════════════════════════════════════════
class Driver:
    """Persistent driver handle - her frame acip kapatmaz"""

    def __init__(self):
        self._handle = None
        self._lock = threading.Lock()

    def _open(self):
        if self._handle is not None:
            return
        CreateFile = ctypes.windll.kernel32.CreateFileW
        CreateFile.restype = wintypes.HANDLE
        h = CreateFile(DEVICE_NAME, 0xC0000000, 0, None, 3, 0, None)
        if h == -1 or h == 0:
            raise OSError("Driver baglantisi kurulamadi")
        self._handle = h

    def move(self, dx, dy):
        """Relative mouse move"""
        if dx == 0 and dy == 0:
            return
        with self._lock:
            try:
                self._open()
                req = NF_MOUSE_REQUEST(int(dx), int(dy), 0)
                ctypes.windll.kernel32.DeviceIoControl(
                    self._handle, IO_SEND_MOUSE_EVENT,
                    ctypes.byref(req), ctypes.sizeof(req),
                    None, 0, ctypes.byref(wintypes.DWORD(0)), None
                )
            except Exception:
                self._handle = None

    def close(self):
        if self._handle:
            ctypes.windll.kernel32.CloseHandle(self._handle)
            self._handle = None


# ═══════════════════════════════════════════════════════
#                   KALMAN FILTER
# ═══════════════════════════════════════════════════════
class KalmanFilter2D:
    """
    2D Kalman filter - pozisyon + hiz takibi.
    Jitter filtreler, hareket yonunu tahmin eder.
    """

    def __init__(self):
        # State: [x, y, vx, vy]
        self.x = np.zeros(4, dtype=np.float64)
        # State covariance
        self.P = np.eye(4, dtype=np.float64) * 100.0
        # Process noise
        self.Q = np.diag([1.0, 1.0, 3.0, 3.0])
        # Measurement noise - dusuk = modele cok guven, yuksek = smooth
        self.R = np.diag([4.0, 4.0])
        # Transition matrix (dt=1 frame)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)
        self.initialized = False

    def reset(self):
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 100.0
        self.initialized = False

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0], self.x[1]

    def update(self, mx, my):
        if not self.initialized:
            self.x[0] = mx
            self.x[1] = my
            self.initialized = True
            return mx, my

        # Predict
        self.predict()

        # Update
        z = np.array([mx, my], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.x[0], self.x[1]

    @property
    def velocity(self):
        """Hedefin hizi (px/frame)"""
        return self.x[2], self.x[3]

    def predicted_position(self, frames_ahead=1):
        """Gelecekteki pozisyon tahmini"""
        px = self.x[0] + self.x[2] * frames_ahead
        py = self.x[1] + self.x[3] * frames_ahead
        return px, py


# ═══════════════════════════════════════════════════════
#                   BEZIER MOVEMENT
# ═══════════════════════════════════════════════════════
def bezier_point(t, p0, p1, p2):
    """Quadratic Bezier - insan eli gibi egri hareket"""
    u = 1.0 - t
    return u * u * p0 + 2 * u * t * p1 + t * t * p2


def generate_bezier_moves(total_dx, total_dy, steps):
    """
    Toplam hareketi Bezier egrisi uzerinde parcalara bol.
    Her parca kucuk bir mouse move olacak.
    """
    if steps < 2:
        return [(int(round(total_dx)), int(round(total_dy)))]

    # Control point - rastgele sapma ekle (insan eli gibi)
    jx = (np.random.random() - 0.5) * abs(total_dx) * BEZIER_JITTER * 0.3
    jy = (np.random.random() - 0.5) * abs(total_dy) * BEZIER_JITTER * 0.3

    cp_x = total_dx * 0.5 + jx
    cp_y = total_dy * 0.5 + jy

    moves = []
    prev_x, prev_y = 0.0, 0.0
    for i in range(1, steps + 1):
        t = i / steps
        bx = bezier_point(t, 0.0, cp_x, total_dx)
        by = bezier_point(t, 0.0, cp_y, total_dy)
        dx = bx - prev_x
        dy = by - prev_y
        ix, iy = int(round(dx)), int(round(dy))
        if ix != 0 or iy != 0:
            moves.append((ix, iy))
        prev_x, prev_y = bx, by

    return moves if moves else [(int(round(total_dx)), int(round(total_dy)))]


# ═══════════════════════════════════════════════════════
#                   TARGET SELECTION
# ═══════════════════════════════════════════════════════
def nms(boxes, scores, iou_thresh=0.45):
    """Non-Maximum Suppression"""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def postprocess(output, frame_w, frame_h):
    """
    YOLO ciktisini isle. NMS + closest-to-center secimi.
    Returns: list of (cx, cy, w, h, score) in frame coords
    """
    preds = output[0]
    if preds.ndim == 3:
        preds = preds[0]
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T

    if preds.shape[1] >= 5:
        scores = np.max(preds[:, 4:], axis=1) if preds.shape[1] > 5 else preds[:, 4]
    else:
        return []

    mask = scores > CONFIDENCE
    filtered = preds[mask]
    filtered_scores = scores[mask]

    if len(filtered) == 0:
        return []

    # NMS
    boxes = filtered[:, :4].copy()
    keep = nms(boxes, filtered_scores)
    filtered = filtered[keep]
    filtered_scores = filtered_scores[keep]

    # Frame koordinatlarina cevir
    results = []
    for i in range(len(filtered)):
        cx = filtered[i][0] / INPUT_SIZE * frame_w
        cy = filtered[i][1] / INPUT_SIZE * frame_h
        w = filtered[i][2] / INPUT_SIZE * frame_w
        h = filtered[i][3] / INPUT_SIZE * frame_h
        results.append((cx, cy, w, h, filtered_scores[i]))

    return results


def select_target(detections, fov_half):
    """Merkeze en yakin hedefi sec"""
    if not detections:
        return None

    best = None
    best_dist = float('inf')

    for (cx, cy, w, h, score) in detections:
        # Head targeting: bbox'in ust %30'u
        aim_y = cy - h * (0.5 - HEAD_OFFSET)
        aim_x = cx

        dist = math.sqrt((aim_x - fov_half) ** 2 + (aim_y - fov_half) ** 2)
        if dist < best_dist:
            best_dist = dist
            best = (aim_x, aim_y, score)

    return best


# ═══════════════════════════════════════════════════════
#                   SCREEN CAPTURE
# ═══════════════════════════════════════════════════════
def get_screen_center():
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0) // 2, user32.GetSystemMetrics(1) // 2


# ═══════════════════════════════════════════════════════
#                  FOV OVERLAY (CIRCLE)
# ═══════════════════════════════════════════════════════
class FOVOverlay:
    """
    Ekranin ortasinda yuvarlak FOV gosterir.
    Win32 layered window - tamamen seffaf, sadece daire gorunur.
    Oyunu engellemez (click-through).
    """

    def __init__(self, center_x, center_y, radius):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.visible = SHOW_FOV
        self._hwnd = None
        self._thread = None
        self._running = False

    def start(self):
        if not self.visible:
            return
        self._running = True
        self._thread = threading.Thread(target=self._create_window, daemon=True)
        self._thread.start()
        time.sleep(0.3)  # Pencerenin olusmasini bekle

    def stop(self):
        self._running = False
        if self._hwnd:
            try:
                ctypes.windll.user32.PostMessageW(self._hwnd, 0x0010, 0, 0)  # WM_CLOSE
            except Exception:
                pass

    def toggle(self):
        """Goster/gizle"""
        self.visible = not self.visible
        if self._hwnd:
            SW_SHOW = 5
            SW_HIDE = 0
            ctypes.windll.user32.ShowWindow(self._hwnd, SW_SHOW if self.visible else SW_HIDE)

    def _create_window(self):
        """Win32 layered transparent window olustur"""
        import ctypes.wintypes

        # Window class
        WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_uint,
                                      ctypes.c_void_p, ctypes.c_void_p)

        def wnd_proc(hwnd, msg, wparam, lparam):
            if msg == 0x000F:  # WM_PAINT
                self._on_paint(hwnd)
                return 0
            if msg == 0x0002:  # WM_DESTROY
                ctypes.windll.user32.PostQuitMessage(0)
                return 0
            return ctypes.windll.user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        self._wnd_proc = WNDPROC(wnd_proc)

        wc = ctypes.wintypes.WNDCLASS()
        wc.lpfnWndProc = self._wnd_proc
        wc.hInstance = ctypes.windll.kernel32.GetModuleHandleW(None)
        wc.lpszClassName = "FOVOverlayClass"
        wc.hbrBackground = 0
        wc.style = 0

        atom = ctypes.windll.user32.RegisterClassW(ctypes.byref(wc))

        # Window size (FOV capi kadar)
        size = self.radius * 2 + 20
        x = self.center_x - size // 2
        y = self.center_y - size // 2

        # WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW
        ex_style = 0x00080000 | 0x00000020 | 0x00000008 | 0x00000080
        # WS_POPUP | WS_VISIBLE
        style = 0x80000000 | 0x10000000

        self._hwnd = ctypes.windll.user32.CreateWindowExW(
            ex_style, "FOVOverlayClass", "FOV",
            style, x, y, size, size,
            None, None, wc.hInstance, None
        )

        # Layered window: siyah rengi seffaf yap
        # LWA_COLORKEY = 0x01, LWA_ALPHA = 0x02
        ctypes.windll.user32.SetLayeredWindowAttributes(
            self._hwnd, 0x00000000, 0, 0x01  # Siyah = seffaf
        )

        # Always on top
        HWND_TOPMOST = -1
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        ctypes.windll.user32.SetWindowPos(
            self._hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE
        )

        # Message loop
        msg = ctypes.wintypes.MSG()
        while self._running:
            ret = ctypes.windll.user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1)
            if ret:
                if msg.message == 0x0012:  # WM_QUIT
                    break
                ctypes.windll.user32.TranslateMessage(ctypes.byref(msg))
                ctypes.windll.user32.DispatchMessageW(ctypes.byref(msg))
            else:
                # Redraw periyodik
                ctypes.windll.user32.InvalidateRect(self._hwnd, None, True)
                time.sleep(0.033)  # ~30fps overlay

    def _on_paint(self, hwnd):
        """Daire ciz"""

        class PAINTSTRUCT(ctypes.Structure):
            _fields_ = [
                ("hdc", ctypes.c_void_p),
                ("fErase", ctypes.c_int),
                ("rcPaint_left", ctypes.c_long),
                ("rcPaint_top", ctypes.c_long),
                ("rcPaint_right", ctypes.c_long),
                ("rcPaint_bottom", ctypes.c_long),
                ("fRestore", ctypes.c_int),
                ("fIncUpdate", ctypes.c_int),
                ("rgbReserved", ctypes.c_byte * 32),
            ]

        ps = PAINTSTRUCT()
        hdc = ctypes.windll.user32.BeginPaint(hwnd, ctypes.byref(ps))

        # Arkaplan siyah (seffaf olacak)
        size = self.radius * 2 + 20

        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

        rect = RECT(0, 0, size, size)
        black_brush = ctypes.windll.gdi32.CreateSolidBrush(0x00000000)
        ctypes.windll.user32.FillRect(hdc, ctypes.byref(rect), black_brush)
        ctypes.windll.gdi32.DeleteObject(black_brush)

        # Daire ciz - kalem (pen) olustur
        r, g, b = FOV_COLOR
        color = r | (g << 8) | (b << 16)  # COLORREF = 0x00BBGGRR
        pen = ctypes.windll.gdi32.CreatePen(0, FOV_THICKNESS, color)  # PS_SOLID=0
        old_pen = ctypes.windll.gdi32.SelectObject(hdc, pen)

        # Ici bos brush (sadece kenar cizgisi)
        null_brush = ctypes.windll.gdi32.GetStockObject(5)  # NULL_BRUSH
        old_brush = ctypes.windll.gdi32.SelectObject(hdc, null_brush)

        # Elips ciz (daire)
        margin = 10
        ctypes.windll.gdi32.Ellipse(hdc, margin, margin,
                                     margin + self.radius * 2,
                                     margin + self.radius * 2)

        # Temizle
        ctypes.windll.gdi32.SelectObject(hdc, old_pen)
        ctypes.windll.gdi32.SelectObject(hdc, old_brush)
        ctypes.windll.gdi32.DeleteObject(pen)

        ctypes.windll.user32.EndPaint(hwnd, ctypes.byref(ps))



# ═══════════════════════════════════════════════════════
#              THREADED INFERENCE ENGINE
# ═══════════════════════════════════════════════════════
class InferenceEngine:
    """
    Ayri thread'de capture + inference yapar.
    Ana thread sadece sonucu okuyup mouse hareket ettirir.
    Boylece inference suresi mouse hareketini engellemez.
    """

    def __init__(self, model_path, center_x, center_y):
        self.center_x = center_x
        self.center_y = center_y
        self.running = False

        # ONNX
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"  Provider: {self.session.get_providers()[0]}")

        # Screen capture
        self.sct = mss.mss()

        # Shared result
        self._lock = threading.Lock()
        self._result = None  # (target_x, target_y, confidence) or None
        self._frame_time = 0.0
        self._fps = 0.0

        # Thread
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    @property
    def result(self):
        with self._lock:
            return self._result

    @property
    def fps(self):
        return self._fps

    def _loop(self):
        half = FOV // 2
        monitor = {
            "left": self.center_x - half,
            "top": self.center_y - half,
            "width": FOV,
            "height": FOV,
        }

        fps_counter = deque(maxlen=30)

        while self.running:
            t0 = time.perf_counter()

            # Capture
            frame = np.array(self.sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

            # Preprocess
            img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
            img = (img.astype(np.float32) / 255.0)
            img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))[np.newaxis])

            # Inference
            outputs = self.session.run(None, {self.input_name: img})

            # Postprocess
            detections = postprocess(outputs, FOV, FOV)
            target = select_target(detections, half)

            with self._lock:
                self._result = target

            # FPS
            dt = time.perf_counter() - t0
            fps_counter.append(dt)
            if len(fps_counter) > 0:
                self._fps = len(fps_counter) / sum(fps_counter)

            # FPS limit
            min_frame_time = 1.0 / MAX_FPS
            elapsed = time.perf_counter() - t0
            if elapsed < min_frame_time:
                time.sleep(min_frame_time - elapsed)


# ═══════════════════════════════════════════════════════
#                     MAIN LOOP
# ═══════════════════════════════════════════════════════
def is_mouse2_pressed():
    return ctypes.windll.user32.GetAsyncKeyState(VK_RBUTTON) & 0x8000 != 0


def main():
    print()
    print("  ╔═══════════════════════════════════╗")
    print("  ║     AIM ASSIST - PRO EDITION      ║")
    print("  ╠═══════════════════════════════════╣")
    print(f"  ║  FOV: {FOV}px  Conf: {CONFIDENCE}          ║")
    print(f"  ║  Smooth: {AIM_SMOOTH}  Deadzone: {DEADZONE}px    ║")
    print(f"  ║  Head%: {int(HEAD_OFFSET*100)}  Predict: {PREDICTION_FACTOR}   ║")
    print(f"  ║  Bezier: {BEZIER_STEPS} steps               ║")
    print("  ║  Aktif: Mouse2 (Sag Tik)         ║")
    print("  ╚═══════════════════════════════════╝")
    print()

    center_x, center_y = get_screen_center()
    print(f"  Ekran: {center_x*2}x{center_y*2} | Merkez: ({center_x},{center_y})")

    # FOV Overlay
    overlay = FOVOverlay(center_x, center_y, FOV // 2)
    overlay.start()
    print(f"  FOV Overlay: {'ACIK' if SHOW_FOV else 'KAPALI'}")

    # Driver
    driver = Driver()
    print("  Driver: OK")

    # Inference engine (ayri thread)
    engine = InferenceEngine(MODEL_PATH, center_x, center_y)
    engine.start()
    print("  Inference thread: Basladi")
    print("  Bekleniyor...\n")

    # Aim state
    kalman = KalmanFilter2D()
    was_active = False
    last_print = 0

    half = FOV // 2
    frame_interval = 1.0 / MAX_FPS

    try:
        while True:
            loop_start = time.perf_counter()

            if not is_mouse2_pressed():
                if was_active:
                    kalman.reset()
                    was_active = False
                time.sleep(0.003)
                continue

            was_active = True

            # Inference sonucunu oku (non-blocking)
            target = engine.result

            if target is None:
                # Hedef yok - Kalman prediction ile devam et (kisa sure)
                if kalman.initialized:
                    px, py = kalman.predict()
                    # Ama fazla uzaklasmadan dur
                    vx, vy = kalman.velocity
                    if abs(vx) < 0.5 and abs(vy) < 0.5:
                        pass  # Duragan, hareket etme
                time.sleep(0.001)
                continue

            target_x, target_y, conf = target

            # Offset: hedefin FOV merkezine uzakligi
            raw_offset_x = target_x - half
            raw_offset_y = target_y - half

            # Kalman filter: jitter filtrele + hiz hesapla
            filtered_x, filtered_y = kalman.update(raw_offset_x, raw_offset_y)

            # Hedef hiz tahmini: hareket eden hedefe ondan git
            vx, vy = kalman.velocity
            pred_x = filtered_x + vx * PREDICTION_FACTOR
            pred_y = filtered_y + vy * PREDICTION_FACTOR

            # Mesafe
            distance = math.sqrt(pred_x ** 2 + pred_y ** 2)

            # Deadzone
            if distance < DEADZONE:
                time.sleep(0.001)
                continue

            # Smoothing: mesafeye gore adaptif
            # Yakinken cok yavas (overshoot onle), uzakken hizli
            t_factor = min(distance / (half * 0.8), 1.0)  # 0..1
            smooth_divisor = AIM_SMOOTH * (1.0 + (1.0 - t_factor) * 2.0)
            move_x = pred_x / smooth_divisor
            move_y = pred_y / smooth_divisor

            # Minimum hareket
            if abs(move_x) < 0.4 and abs(move_y) < 0.4:
                time.sleep(0.001)
                continue

            # Bezier curve ile parcali gonder (smooth + human-like)
            steps = BEZIER_STEPS if distance > 20 else 2
            moves = generate_bezier_moves(move_x, move_y, steps)
            for dx, dy in moves:
                driver.move(dx, dy)
                if len(moves) > 1:
                    time.sleep(0.001)  # Parcalar arasi kucuk bekleme

            # FPS / debug print
            now = time.perf_counter()
            if now - last_print > 2.0:
                last_print = now
                print(f"  [FPS:{engine.fps:.0f}] conf:{conf:.2f} "
                      f"offset:({filtered_x:.1f},{filtered_y:.1f}) "
                      f"vel:({vx:.1f},{vy:.1f}) dist:{distance:.1f}")

            # Frame rate limit
            elapsed = time.perf_counter() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    except KeyboardInterrupt:
        print("\n  Kapatiliyor...")
    finally:
        engine.stop()
        overlay.stop()
        driver.close()
        print("  Kapandi.")


if __name__ == "__main__":
    main()

