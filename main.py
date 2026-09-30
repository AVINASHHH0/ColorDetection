import cv2
import numpy as np
import time

# ==============================
# CAMERA
# ==============================

# Put your current phone camera URL here
cap = cv2.VideoCapture("http://10.124.21.157:8080/stream.mjpg")

# ==============================
# COLOR RANGES
# ==============================

colors = {

    "RED": (
        [(0, 120, 70), (10, 255, 255),
         (170, 120, 70), (179, 255, 255)],
        (0, 0, 255)
    ),

    "ORANGE": (
        [(10, 120, 70), (25, 255, 255)],
        (0, 165, 255)
    ),

    "YELLOW": (
        [(25, 100, 100), (35, 255, 255)],
        (0, 255, 255)
    ),

    "GREEN": (
        [(35, 50, 50), (85, 255, 255)],
        (0, 255, 0)
    ),

    "CYAN": (
        [(85, 80, 50), (100, 255, 255)],
        (255, 255, 0)
    ),

    "BLUE": (
        [(100, 100, 50), (130, 255, 255)],
        (255, 0, 0)
    ),

    "PURPLE": (
        [(130, 70, 50), (155, 255, 255)],
        (255, 0, 255)
    ),

    "PINK": (
        [(155, 70, 50), (179, 255, 255)],
        (203, 192, 255)
    ),

    "BROWN": (
        [(5, 80, 30), (20, 255, 200)],
        (42, 42, 165)
    ),

    "WHITE": (
        [(0, 0, 180), (179, 70, 255)],
        (255, 255, 255)
    ),

    "GRAY": (
        [(0, 0, 50), (179, 60, 180)],
        (128, 128, 128)
    ),

    "BLACK": (
        [(0, 0, 0), (179, 255, 50)],
        (50, 50, 50)
    )
}

# ==============================
# FPS
# ==============================

prev_time = time.time()

# ==============================
# MAIN LOOP
# ==============================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera not found")
        break

    # Resize for better performance
    frame = cv2.resize(frame, (960, 540))

    # Convert BGR → HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    detected_colors = set()
    total_objects = 0

    # ==============================
    # DETECT EACH COLOR
    # ==============================

    for name, (ranges, box_color) in colors.items():

        # Special case for RED because red wraps around HSV
        if name == "RED":

            mask1 = cv2.inRange(
                hsv,
                np.array(ranges[0]),
                np.array(ranges[1])
            )

            mask2 = cv2.inRange(
                hsv,
                np.array(ranges[2]),
                np.array(ranges[3])
            )

            mask = cv2.bitwise_or(mask1, mask2)

        else:

            lower = np.array(ranges[0])
            upper = np.array(ranges[1])

            mask = cv2.inRange(hsv, lower, upper)

        # Remove small noise
        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # Find objects
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # ==============================
        # PROCESS OBJECTS
        # ==============================

        for contour in contours:

            area = cv2.contourArea(contour)

            # Ignore tiny objects
            if area > 500:

                x, y, w, h = cv2.boundingRect(contour)

                # Center coordinates
                center_x = x + w // 2
                center_y = y + h // 2

                # Bounding box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    box_color,
                    2
                )

                # Center point
                cv2.circle(
                    frame,
                    (center_x, center_y),
                    5,
                    (255, 255, 255),
                    -1
                )

                # Color name
                cv2.putText(
                    frame,
                    name,
                    (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2
                )

                # Coordinates
                cv2.putText(
                    frame,
                    f"Center: ({center_x},{center_y})",
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1
                )

                total_objects += 1
                detected_colors.add(name)

    # ==============================
    # FPS
    # ==============================

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    # ==============================
    # INFORMATION PANEL
    # ==============================

    cv2.rectangle(
        frame,
        (10, 10),
        (500, 100),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Objects: {total_objects}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    detected_text = ", ".join(detected_colors)

    cv2.putText(
        frame,
        f"Colors: {detected_text}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    # ==============================
    # SHOW CAMERA
    # ==============================

    cv2.imshow(
        "Multi-Color Detection",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==============================
# RELEASE
# ==============================

cap.release()
cv2.destroyAllWindows()