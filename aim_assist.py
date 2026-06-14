"""
Python Aim Assist - Tek Dosya
Driver üzerinden mouse hareketi, ONNX model ile hedef tespiti.
Mouse2 (sağ tık) basılı tutunca aktif olur.
"""

import ctypes
import ctypes.wintypes as wintypes
import time
import random
import os
import cv2
import numpy as np
import mss
import onnxruntime as ort

# ╔══════════════════════════════════════════╗
# ║           AYARLAR (CONFIG)               ║
# ╠══════════════════════════════════════════╣
MODEL_PATH = os.path.abspath("1.5kR6.onnx")
FOV = 320                    # Ekranın ortasından kaç piksel alan taranacak (kare)
CONFIDENCE_THRESHOLD = 0.45  # Minimum güven eşiği (0.0 - 1.0)
AIM_SPEED = 0.6              # Aim hızı (0.1 yavaş - 1.0 anlık)
INPUT_SIZE = 640             # ONNX model giriş boyutu
# ╚══════════════════════════════════════════╝

# === [ DRIVER AYARLARI ] ===
DEVICE_NAME = r"\\.\volmgra"
IO_SEND_MOUSE_EVENT = 0x1FE3BD28

# === [ Win32 Sabitler ] ===
VK_RBUTTON = 0x02  # Sağ tık (Mouse2)


# === [ Yapılar ] ===
class MouseFlags:
    LeftButtonDown = 1
    LeftButtonUp = 2


class NF_MOUSE_REQUEST(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("ButtonFlags", ctypes.c_short),
    ]


# === [ Driver Fonksiyonları ] ===
def open_driver():
    CreateFile = ctypes.windll.kernel32.CreateFileW
    CreateFile.restype = wintypes.HANDLE
    handle = CreateFile(DEVICE_NAME, 0xC0000000, 0, None, 3, 0, None)
    if handle == -1 or handle == 0:
        raise OSError("Driver handle alinamadi.")
    return handle


def send_mouse_move(x, y):
    try:
        hDriver = open_driver()
        req = NF_MOUSE_REQUEST(int(x), int(y), 0)
        ctypes.windll.kernel32.DeviceIoControl(
            hDriver, IO_SEND_MOUSE_EVENT,
            ctypes.byref(req), ctypes.sizeof(req),
            None, 0, ctypes.byref(wintypes.DWORD(0)), None
        )
        ctypes.windll.kernel32.CloseHandle(hDriver)
    except Exception as e:
        print(f"Move gonderilemedi: {e}")


def send_click():
    try:
        hDriver = open_driver()
        req = NF_MOUSE_REQUEST(0, 0, MouseFlags.LeftButtonDown)
        ctypes.windll.kernel32.DeviceIoControl(
            hDriver, IO_SEND_MOUSE_EVENT,
            ctypes.byref(req), ctypes.sizeof(req),
            None, 0, ctypes.byref(wintypes.DWORD(0)), None
        )
        time.sleep(random.uniform(0.05, 0.15))
        req.ButtonFlags = MouseFlags.LeftButtonUp
        ctypes.windll.kernel32.DeviceIoControl(
            hDriver, IO_SEND_MOUSE_EVENT,
            ctypes.byref(req), ctypes.sizeof(req),
            None, 0, ctypes.byref(wintypes.DWORD(0)), None
        )
        ctypes.windll.kernel32.CloseHandle(hDriver)
    except Exception as e:
        print(f"Click gonderilemedi: {e}")


# === [ Mouse2 (Sağ Tık) Kontrol ] ===
def is_mouse2_pressed():
    return ctypes.windll.user32.GetAsyncKeyState(VK_RBUTTON) & 0x8000 != 0


# === [ ONNX Model Yükleme ] ===
def load_model():
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    print(f"Model yukleniyor: {MODEL_PATH}")
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    print(f"Aktif provider: {session.get_providers()}")
    return session


# === [ Ekran Yakalama ] ===
def get_screen_center():
    user32 = ctypes.windll.user32
    w = user32.GetSystemMetrics(0)
    h = user32.GetSystemMetrics(1)
    return w // 2, h // 2


def capture_fov(sct, center_x, center_y):
    half = FOV // 2
    monitor = {
        "left": center_x - half,
        "top": center_y - half,
        "width": FOV,
        "height": FOV,
    }
    frame = np.array(sct.grab(monitor))
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


# === [ Model Inference ] ===
def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)    # Batch boyutu ekle
    return img


def postprocess(output, frame_width, frame_height):
    """
    YOLOv5/v8 çıktısını işle. En yüksek confidence'lı hedefi döndür.
    output shape: (1, N, 5+classes) veya (1, 5+classes, N)
    """
    preds = output[0]

    # Eğer shape (1, 5+C, N) ise transpose et
    if preds.ndim == 3:
        preds = preds[0]
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T

    # Her satır: [cx, cy, w, h, conf, ...class_scores] veya [cx, cy, w, h, ...class_scores]
    # YOLOv8 formatı: objectness yok, class score direkt
    if preds.shape[1] == 5:
        # [cx, cy, w, h, score]
        scores = preds[:, 4]
    elif preds.shape[1] > 5:
        # YOLOv5: [cx, cy, w, h, obj_conf, class1, class2, ...]
        # YOLOv8: [cx, cy, w, h, class1, class2, ...]
        # Basitçe: 4. indexten sonraki en yüksek değeri al
        scores = np.max(preds[:, 4:], axis=1)
    else:
        return None

    # Confidence filtresi
    mask = scores > CONFIDENCE_THRESHOLD
    filtered = preds[mask]
    filtered_scores = scores[mask]

    if len(filtered) == 0:
        return None

    # En yüksek confidence
    best_idx = np.argmax(filtered_scores)
    best = filtered[best_idx]

    cx = best[0] / INPUT_SIZE * frame_width
    cy = best[1] / INPUT_SIZE * frame_height

    return cx, cy, filtered_scores[best_idx]


# === [ Ana Döngü ] ===
def main():
    print("=" * 50)
    print("  PYTHON AIM ASSIST")
    print("=" * 50)
    print(f"  FOV: {FOV}px | Confidence: {CONFIDENCE_THRESHOLD}")
    print(f"  Aim Speed: {AIM_SPEED} | Model: {MODEL_PATH}")
    print(f"  Aktivasyon: Mouse2 (Sag Tik) basili tut")
    print("=" * 50)

    session = load_model()
    input_name = session.get_inputs()[0].name

    center_x, center_y = get_screen_center()
    print(f"  Ekran merkezi: ({center_x}, {center_y})")
    print("  Calistiriliyor... (Ctrl+C ile kapat)")
    print("=" * 50)

    sct = mss.mss()

    while True:
        try:
            # Mouse2 basılı değilse bekle
            if not is_mouse2_pressed():
                time.sleep(0.005)
                continue

            # Ekran yakala
            frame = capture_fov(sct, center_x, center_y)

            # Preprocess
            input_tensor = preprocess(frame)

            # Inference
            outputs = session.run(None, {input_name: input_tensor})

            # Postprocess
            result = postprocess(outputs, FOV, FOV)

            if result is None:
                continue

            target_x, target_y, conf = result

            # Hedefin ekran merkezine göre offset'i
            half = FOV // 2
            offset_x = target_x - half
            offset_y = target_y - half

            # Aim speed uygula (smoothing)
            move_x = offset_x * AIM_SPEED
            move_y = offset_y * AIM_SPEED

            # Driver ile mouse hareket ettir
            if abs(move_x) > 1 or abs(move_y) > 1:
                send_mouse_move(int(move_x), int(move_y))

        except KeyboardInterrupt:
            print("\nKapatiliyor...")
            break
        except Exception as e:
            print(f"Hata: {e}")
            time.sleep(0.1)


if __name__ == "__main__":
    main()
