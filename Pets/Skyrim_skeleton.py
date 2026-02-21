import cv2
import mediapipe as mp
import time
import os
from pathlib import Path

VIDEO_PATH = Path("C:\\Users\\alias\\Downloads\\skyrim-skeleton.mp4")

video_opened = False


def play_video(video_path: Path):
    global video_opened
    if not video_opened:
        os.startfile(video_path)
        video_opened = True


def draw_warning(frame, text="doomscrolling alarm"):
    h, w = frame.shape[:2]
    box_w, box_h = 520, 70
    x1 = (w - box_w) // 2
    y1 = 24
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 0, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 160), 3)

    cv2.putText(
        frame,
        text.upper(),
        (x1 + 24, y1 + 48),
        cv2.FONT_HERSHEY_DUPLEX,
        1.1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def iris_ratio(iris_y, upper_y, lower_y):
    return (iris_y - upper_y) / ((lower_y - upper_y) + 1e-6)


def main():
    timer = 2.0
    start_threshold = 0.52
    stop_threshold = 0.42

    if not VIDEO_PATH.exists():
        print("Уебан у тебя нет камеры")
        return

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(refine_landmarks=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Даун открой камеру")
        return

    doomscroll_start = None
    looking_down = False

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            now = time.time()

            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark

                left_upper = lm[159]
                left_lower = lm[145]
                right_upper = lm[386]
                right_lower = lm[374]

                left_iris = lm[468]
                right_iris = lm[473]

                l_ratio = iris_ratio(left_iris.y, left_upper.y, left_lower.y)
                r_ratio = iris_ratio(right_iris.y, right_upper.y, right_lower.y)
                avg_ratio = (l_ratio + r_ratio) / 2

                if looking_down:
                    looking_down = avg_ratio > stop_threshold
                else:
                    looking_down = avg_ratio > start_threshold

                if looking_down:
                    if doomscroll_start is None:
                        doomscroll_start = now

                    if (now - doomscroll_start) >= timer:
                        play_video(VIDEO_PATH)
                else:
                    doomscroll_start = None

                cv2.putText(
                    frame,
                    f"ratio: {avg_ratio:.3f}",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                doomscroll_start = None

            if doomscroll_start and not video_opened:
                draw_warning(frame)

            cv2.imshow("lock in", frame)
            if cv2.waitKey(1) == 27:
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()