import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math
import sys
import threading

# --- Configuration ---
WEBCAM_INDEX = 0        # Default webcam is usually 0. Change if you have multiple.
FRAME_WIDTH = 640       # Width of the camera frame for processing
FRAME_HEIGHT = 480      # Height of the camera frame for processing

# --- Mapping Configuration ---
X_MAP_RANGE = (0.15, 0.85) # Use 15% to 85% of camera width for mapping
Y_MAP_RANGE = (0.15, 0.85) # Use 15% to 85% of camera height for mapping

# --- Movement Sensitivity / Acceleration (Refined) ---
BASE_SENSITIVITY = 1.0      # Multiplier for movement *below* threshold. 1.0 aims for 1:1 feeling.
SPEED_THRESHOLD = 10.0      # Hand movement speed (pixels/frame diff) below which only BASE_SENSITIVITY applies. Tune this!
ACCELERATION_FACTOR = 0.015 # How much speed *above threshold* affects acceleration. Note: Scales non-linearly now. Tune this!
ACCEL_POWER = 1.5           # Power for non-linear acceleration (1.0=linear, 1.5-2.0=curves). Tune this!
MAX_ACCEL_SCALE = 5.0       # Optional: Maximum acceleration multiplier

# --- Smoothing ---
# Increased smoothing helps manage acceleration curve and tracking noise
SMOOTHING = 9

# --- Dwell Click Configuration ---
DWELL_TIME = 0.8
DWELL_RADIUS = 15
CLICK_COOLDOWN = 0.7

# --- Overlay Visual Configuration ---
CURSOR_COLOR = (0, 255, 0)
CURSOR_RADIUS = 8
CURSOR_CLICK_COLOR = (0, 0, 255)
CURSOR_CLICK_DURATION = 0.1

# --- Global Variables ---
g_target_x, g_target_y = 0, 0
g_screen_width, g_screen_height = pyautogui.size()
g_tracking_active = True
g_last_click_time_overlay = 0
g_lock = threading.Lock()

# --- Hand Tracking Function (Revised for Threshold/Non-Linear Acceleration) ---
def hand_tracking_worker():
    global g_target_x, g_target_y, g_tracking_active, g_last_click_time_overlay, g_lock, g_screen_width, g_screen_height

    print("Hand tracking thread started (Threshold/Non-Linear Accel & Dwell).")
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam index {WEBCAM_INDEX}")
        with g_lock: g_tracking_active = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    prev_x, prev_y = g_screen_width // 2, g_screen_height // 2
    pyautogui.moveTo(int(prev_x), int(prev_y))

    last_click_time_action = 0
    dwell_start_time = 0
    dwell_pos_start = (int(prev_x), int(prev_y))
    is_dwelling = False

    while True:
        with g_lock:
            if not g_tracking_active: break

        success, image = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # 1. Calculate Raw Target Screen Position
                target_x_raw = np.interp(index_tip.x, X_MAP_RANGE, (0, g_screen_width))
                target_y_raw = np.interp(index_tip.y, Y_MAP_RANGE, (0, g_screen_height))
                target_x_raw = np.clip(target_x_raw, 0, g_screen_width)
                target_y_raw = np.clip(target_y_raw, 0, g_screen_height)

                # 2. Calculate Difference & Speed
                # Difference from the *previous smoothed position*
                diff_x = target_x_raw - prev_x
                diff_y = target_y_raw - prev_y
                speed = math.sqrt(diff_x**2 + diff_y**2)

                # 3. Calculate Acceleration Scale (Threshold + Non-Linear)
                if speed < SPEED_THRESHOLD:
                    # Below threshold: Use only base sensitivity
                    scale = BASE_SENSITIVITY
                else:
                    # Above threshold: Apply non-linear acceleration
                    # Calculate speed *above* the threshold
                    effective_speed = speed - SPEED_THRESHOLD
                    # Apply non-linear scaling
                    accel_component = ACCELERATION_FACTOR * (effective_speed ** ACCEL_POWER)
                    scale = BASE_SENSITIVITY + accel_component
                    # Apply optional cap
                    if MAX_ACCEL_SCALE > 0:
                        scale = min(scale, MAX_ACCEL_SCALE)

                # 4. Calculate Smoothed Step with Acceleration Scale
                # Apply the calculated scale to the difference, then smooth
                step_x = (diff_x * scale) / SMOOTHING
                step_y = (diff_y * scale) / SMOOTHING

                # 5. Calculate New Smoothed Position
                curr_x = prev_x + step_x
                curr_y = prev_y + step_y

                # Clamp final position
                curr_x = np.clip(curr_x, 0, g_screen_width)
                curr_y = np.clip(curr_y, 0, g_screen_height)

                current_pos = (int(curr_x), int(curr_y))

                # 6. Update Global Coordinates & Move Mouse
                with g_lock:
                    g_target_x = current_pos[0]
                    g_target_y = current_pos[1]

                if abs(current_pos[0] - int(prev_x)) > 0 or abs(current_pos[1] - int(prev_y)) > 0:
                     pyautogui.moveTo(current_pos[0], current_pos[1], _pause=False) # Use _pause=False for responsiveness

                # 7. Dwell Click Logic (Uses final smoothed `current_pos`)
                current_time = time.time()
                move_distance = math.sqrt((current_pos[0] - dwell_pos_start[0])**2 + (current_pos[1] - dwell_pos_start[1])**2)

                if not is_dwelling:
                    dwell_start_time = current_time
                    dwell_pos_start = current_pos # Start dwell check from current smoothed pos
                    is_dwelling = True
                elif move_distance > DWELL_RADIUS:
                    is_dwelling = False
                else:
                    if is_dwelling and (current_time - dwell_start_time) > DWELL_TIME:
                        if current_time - last_click_time_action > CLICK_COOLDOWN:
                            print(f"Dwell Click Action at {dwell_pos_start}!") # Click where dwell started
                            pyautogui.click(dwell_pos_start[0], dwell_pos_start[1], _pause=False)
                            last_click_time_action = current_time
                            is_dwelling = False
                            with g_lock:
                               g_last_click_time_overlay = current_time
                        # else: Cooldown active

                # 8. Update Previous Position for Next Frame
                prev_x, prev_y = curr_x, curr_y
                # End landmark loop
            # End if landmarks
        else: # No hands detected
             if is_dwelling: is_dwelling = False

        time.sleep(0.001) # Yield

    # --- Cleanup ---
    print("Releasing webcam...")
    cap.release()
    hands.close()
    print("Hand tracking thread finished.")


# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting Touch Simulator with Overlay (Threshold Accel & Dwell)...")
    print(f"Screen: {g_screen_width}x{g_screen_height}")
    print(f"Sensitivity: Base={BASE_SENSITIVITY}, Threshold={SPEED_THRESHOLD}px/frame")
    print(f"Accel: Factor={ACCELERATION_FACTOR}, Power={ACCEL_POWER} (Above Threshold)")
    print(f"Smoothing: {SMOOTHING}")
    print(f"Dwell: Time={DWELL_TIME}s, Radius={DWELL_RADIUS}px")
    print("Move index finger. Slow=precise, Fast=jump. Hold still to click.")
    print("Starting hand tracking thread...")

    tracking_thread = threading.Thread(target=hand_tracking_worker, daemon=True)
    tracking_thread.start()

    print("Creating overlay window...")
    print("Press 'q' in overlay window or Ctrl+C in console to quit.")

    # --- OpenCV Overlay ---
    overlay_img = np.zeros((g_screen_height, g_screen_width, 3), dtype=np.uint8)
    window_name = "Touch Overlay (Press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # --- Main Overlay Loop ---
    while True:
        with g_lock:
            if not g_tracking_active: break
            current_x, current_y = g_target_x, g_target_y
            last_click_viz_time = g_last_click_time_overlay

        is_clicking_now = time.time() - last_click_viz_time < CURSOR_CLICK_DURATION
        overlay_img = np.zeros((g_screen_height, g_screen_width, 3), dtype=np.uint8) # Clear
        color = CURSOR_CLICK_COLOR if is_clicking_now else CURSOR_COLOR

        draw_x = np.clip(current_x, 0, g_screen_width - 1)
        draw_y = np.clip(current_y, 0, g_screen_height - 1)
        cv2.circle(overlay_img, (draw_x, draw_y), CURSOR_RADIUS, color, -1)

        cv2.imshow(window_name, overlay_img)

        key = cv2.waitKey(15) # Reduce overlay update rate slightly if needed (e.g., 15-20ms)
        if key & 0xFF == ord('q'):
            print("Quit key pressed.")
            with g_lock: g_tracking_active = False
            break
        if not tracking_thread.is_alive():
             print("Tracking thread stopped.")
             with g_lock: g_tracking_active = False
             break

    # --- Cleanup ---
    print("Shutting down...")
    with g_lock: g_tracking_active = False

    if tracking_thread.is_alive():
        print("Waiting for tracking thread...")
        tracking_thread.join(timeout=1.5)
        if tracking_thread.is_alive(): print("Warning: Tracking thread did not terminate.")

    cv2.destroyAllWindows()
    print("Done.")
    sys.exit()