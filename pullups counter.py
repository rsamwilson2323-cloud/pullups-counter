import cv2
import mediapipe as mp

# ---------------- INIT ----------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Fullscreen window
win = "Pull-Ups Counter"
cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)

count = 0
position = None

# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark

        # Nose position (to place count)
        nose = lm[mp_pose.PoseLandmark.NOSE]
        cx, cy = int(nose.x * w), int(nose.y * h)

        # Pull-up logic
        if nose.y < 0.4:
            if position == "down":
                count += 1
                position = "up"
        elif nose.y > 0.6:
            position = "down"

        # Draw pose
        mp_draw.draw_landmarks(
            frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # Draw count near head
        cv2.putText(frame, str(count),
                    (cx - 20, cy - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    5)

    cv2.imshow(win, frame)

    # ENTER key exits
    if cv2.waitKey(1) == 13:
        break

# ---------------- EXIT ----------------
cap.release()
cv2.destroyAllWindows()
