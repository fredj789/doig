#!/usr/bin/env python3
"""Compte les doigts tendus sur une ou deux mains filmées par la webcam."""

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
FINGERS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
           (13, 14, 15, 16), (17, 18, 19, 20))
CONNECTIONS = ((0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17)) + tuple(
    edge for finger in FINGERS for edge in zip(finger, finger[1:])
)


def download_model(path):
    if path.is_file():
        return
    print("Téléchargement du modèle MediaPipe…")
    with urlopen(MODEL_URL, timeout=30) as response:
        data = response.read()
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        temporary_path.write_bytes(data)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def angle(a, b, c):
    """Mesure l'angle ABC en degrés."""
    u = tuple(x - y for x, y in zip(a, b))
    v = tuple(x - y for x, y in zip(c, b))
    length = math.dist(a, b) * math.dist(c, b)
    if length == 0:
        return 0.0
    cosine = sum(x * y for x, y in zip(u, v)) / length
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def projection(point, origin, axis_start, axis_end):
    """Projette le déplacement origin → point sur un axe unitaire."""
    length = math.dist(axis_start, axis_end)
    if length == 0:
        return 0.0
    return sum((p - o) * (b - a) for p, o, a, b
               in zip(point, origin, axis_start, axis_end)) / length


def raised_fingers(points, previous=None):
    """Évalue l'extension dans le repère 3D de la main."""
    if previous is None:
        previous = (False,) * 5

    palm_width = math.dist(points[5], points[17])
    # Un pouce droit peut rester rabattu sur la paume. Sa pointe doit
    # aussi dépasser le bord de l'index, du côté extérieur de la main.
    thumb_out = projection(points[4], points[5], points[17], points[5])
    thumb_reach = math.dist(points[2], points[4])
    thumb_length = math.dist(points[2], points[3]) + math.dist(points[3], points[4])
    thumb = (
        palm_width > 0
        and thumb_length > 0
        and thumb_out > palm_width * (0.03 if previous[0] else 0.08)
        and thumb_reach / thumb_length > (0.85 if previous[0] else 0.90)
    )
    states = [thumb]

    for index, (base, middle, distal, tip) in enumerate(FINGERS[1:], start=1):
        proximal_angle = angle(points[base], points[middle], points[distal])
        distal_angle = angle(points[middle], points[distal], points[tip])
        finger_length = sum(math.dist(points[a], points[b])
                            for a, b in ((base, middle), (middle, distal), (distal, tip)))
        # Un doigt peut rester assez droit tout en étant rabattu depuis sa base.
        forward = projection(points[tip], points[base], points[0], points[base])
        min_angle = 116 if previous[index] else 128
        min_extension = 0.31 if previous[index] else 0.43
        states.append(
            finger_length > 0
            and min(proximal_angle, distal_angle) > min_angle
            and proximal_angle + distal_angle > (270 if previous[index] else 288)
            and forward > min_extension * finger_length
        )
    return states


@dataclass
class StableGesture:
    """Valide une nouvelle position après 120 ms de détection continue."""

    shown: tuple[bool, ...] | None = None
    candidate: tuple[bool, ...] | None = None
    since: float = 0.0
    last_seen: float = 0.0

    def update(self, states, now):
        if now - self.last_seen > 0.25:
            self.shown = self.candidate = None
        self.last_seen = now
        states = tuple(states)
        if states != self.candidate:
            self.candidate, self.since = states, now
        elif now - self.since >= 0.12:
            self.shown = states
        return self.shown


def draw_text(cv2, frame, label, position, size=0.55, color=(238, 242, 249), weight=1):
    cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                size, color, weight, cv2.LINE_AA)


def draw_status(cv2, frame, hands, fps):
    width = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (width, 65), (38, 30, 23), -1)
    cv2.line(frame, (0, 65), (width, 65), (80, 66, 52), 1)
    draw_text(cv2, frame, "COMPTEUR DE DOIGTS", (16, 24), 0.48, (207, 188, 166), 1)
    if not hands:
        status = "Montre une main"
    elif any(states is None for states in hands):
        status = "Analyse du geste..."
    else:
        total = sum(sum(states) for states in hands)
        status = f"{total} / {5 * len(hands)} doigts"
    draw_text(cv2, frame, status, (16, 51), 0.7, weight=2)

    label = f"{fps:.0f} FPS"
    label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    draw_text(cv2, frame, label, (width - label_width - 16, 24), 0.5)
    if width >= 340:
        draw_text(cv2, frame, "Q / ESC", (width - 78, 50), 0.42, (207, 188, 166))


def draw_hand(cv2, frame, pixels, states, color):
    for a, b in CONNECTIONS:
        cv2.line(frame, pixels[a], pixels[b], color, 2, cv2.LINE_AA)
    for pixel in pixels:
        cv2.circle(frame, pixel, 3, (247, 248, 252), -1, cv2.LINE_AA)

    height, width = frame.shape[:2]
    point_x, point_y = pixels[0]
    x = min(max(4, point_x - 61), width - 128)
    y = min(max(72, point_y + 10), height - 34)
    cv2.rectangle(frame, (x, y), (x + 123, y + 29), (38, 30, 23), -1)
    count = "-" if states is None else str(sum(states))
    draw_text(cv2, frame, count, (x + 9, y + 21), 0.6, weight=2)
    for finger in range(5):
        center = (x + 37 + 17 * finger, y + 15)
        dot_color = color if states is not None and states[finger] else (107, 94, 82)
        cv2.circle(frame, center, 4, dot_color, -1, cv2.LINE_AA)


def run(camera_index, hand_limit):
    import cv2
    import mediapipe as mp

    model_path = Path(__file__).resolve().with_name("hand_landmarker.task")
    download_model(model_path)
    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=hand_limit,
    )
    window = "Compteur de doigts"
    camera = cv2.VideoCapture(camera_index)
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Impossible d’ouvrir la caméra {camera_index}.")
        with vision.HandLandmarker.create_from_options(options) as detector:
            start = previous = time.monotonic()
            timestamp = -1
            fps = 0.0
            gestures = {}
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Impossible de lire une image de la caméra.")
                frame = cv2.flip(frame, 1)
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timestamp = max(timestamp + 1, int((time.monotonic() - start) * 1000))
                result = detector.detect_for_video(image, timestamp)

                now = time.monotonic()
                height, width = frame.shape[:2]
                hands = []
                seen = set()
                for landmarks, world, handedness in zip(
                    result.hand_landmarks, result.hand_world_landmarks, result.handedness
                ):
                    label = handedness[0].category_name
                    key = (label, 0)
                    if key in seen:
                        key = (label, 1)
                    seen.add(key)
                    pixels = [(int(p.x * width), int(p.y * height)) for p in landmarks]
                    points = [(p.x, p.y, p.z) for p in world]
                    gesture = gestures.setdefault(key, StableGesture())
                    states = gesture.update(raised_fingers(points, gesture.shown), now)
                    hands.append(states)
                    color = (124, 226, 68) if label == "Right" else (255, 175, 66)
                    draw_hand(cv2, frame, pixels, states, color)

                instant_fps = 1 / max(now - previous, 0.000001)
                fps = 0.85 * fps + 0.15 * instant_fps if fps else instant_fps
                previous = now
                draw_status(cv2, frame, hands, fps)
                cv2.imshow(window, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                try:
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Numéro de la webcam (défaut : 0)")
    parser.add_argument("--mains", type=int, choices=(1, 2), default=2,
                        help="Nombre maximal de mains (défaut : 2)")
    args = parser.parse_args()
    try:
        run(args.camera, args.mains)
    except ImportError as error:
        parser.exit(1, f"Dépendance manquante : {error}\nInstalle avec : python -m pip install mediapipe opencv-python\n")
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"Erreur : {error}\n")


if __name__ == "__main__":
    main()
