import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math
import json
import os
from datetime import datetime

class GestureAid:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        self.setup_directories()
        self.user_profile = None
        self.tracking_active = True
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Tracking state
        self.prev_x, self.prev_y = self.screen_width // 2, self.screen_height // 2
        self.dragging = False
        self.current_gesture = "none"
        self.gesture_start_time = 0
        
        print("🎯 GestureAid Accessibility System Initialized")
    
    def setup_directories(self):
        """Create data directories"""
        os.makedirs("GestureAid_Data", exist_ok=True)
    
    def calculate_distance(self, point1, point2):
        """Calculate distance between two landmarks"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def create_user_profile(self):
        """Simple profile setup"""
        print("\n" + "="*50)
        print("        GESTUREAID SETUP WIZARD")
        print("="*50)
        
        name = input("Enter your name: ").strip() or "User"
        
        print("\nSelect your mobility level:")
        print("1. High - Good hand control")
        print("2. Medium - Some limitations")  
        print("3. Low - Significant limitations")
        
        choice = input("Enter choice (1-3): ").strip()
        mobility_map = {"1": "high", "2": "medium", "3": "low"}
        mobility_level = mobility_map.get(choice, "medium")
        
        # Set parameters based on mobility level
        if mobility_level == "high":
            self.dwell_time = 0.5
            self.sensitivity = 1.2
            self.smoothing = 5
        elif mobility_level == "medium":
            self.dwell_time = 1.0
            self.sensitivity = 1.0
            self.smoothing = 7
        else:  # low
            self.dwell_time = 1.5
            self.sensitivity = 0.8
            self.smoothing = 10
        
        print(f"\n✅ Profile created for {name}")
        print(f"   Mobility: {mobility_level}")
        print(f"   Dwell time: {self.dwell_time}s")
        print(f"   Sensitivity: {self.sensitivity}")
        
        return name, mobility_level
    
    def detect_gesture(self, hand_landmarks):
        """Detect hand gestures"""
        landmarks = hand_landmarks.landmark
        mp_hands = self.mp_hands
        
        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
        wrist = landmarks[mp_hands.HandLandmark.WRIST]
        
        # Calculate distances
        thumb_index_dist = self.calculate_distance(thumb_tip, index_tip)
        
        # Gesture 1: Pinch (Left Click)
        if thumb_index_dist < 0.05:
            return "pinch"
        
        # Gesture 2: Fist (Drag)
        elif self.is_fist(landmarks, wrist):
            return "fist"
        
        # Gesture 3: Peace Sign (Right Click)
        elif (index_tip.y < middle_tip.y and 
              middle_tip.y < ring_tip.y and
              thumb_index_dist > 0.1):
            return "peace"
        
        # Gesture 4: Thumbs Up (Special Action)
        elif (thumb_index_dist > 0.15 and 
              thumb_tip.y < index_tip.y):
            return "thumbs_up"
        
        return "none"
    
    def is_fist(self, landmarks, wrist):
        """Check if hand is making a fist"""
        for finger_tip in [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]:
            dist = self.calculate_distance(landmarks[finger_tip], wrist)
            if dist > 0.25:
                return False
        return True
    
    def execute_action(self, gesture, cursor_pos):
        """Execute action based on gesture"""
        if gesture == "pinch":
            pyautogui.click(cursor_pos[0], cursor_pos[1])
            print("✅ Left click")
            
        elif gesture == "peace":
            pyautogui.rightClick(cursor_pos[0], cursor_pos[1])
            print("✅ Right click")
            
        elif gesture == "fist":
            if not self.dragging:
                pyautogui.mouseDown(cursor_pos[0], cursor_pos[1])
                self.dragging = True
                print("🎯 Drag started")
            else:
                pyautogui.mouseUp()
                self.dragging = False
                print("🎯 Drag ended")
                
        elif gesture == "thumbs_up":
            # Open browser as example special action
            pyautogui.hotkey('win', 'r')
            time.sleep(0.5)
            pyautogui.write('chrome')
            pyautogui.press('enter')
            print("🌐 Opening browser")
    
    def calculate_cursor_position(self, index_tip):
        """Calculate smooth cursor position"""
        # Map hand position to screen
        target_x = np.interp(index_tip.x, (0.15, 0.85), (0, self.screen_width))
        target_y = np.interp(index_tip.y, (0.15, 0.85), (0, self.screen_height))
        
        # Calculate movement
        diff_x = target_x - self.prev_x
        diff_y = target_y - self.prev_y
        
        # Apply sensitivity and smoothing
        step_x = (diff_x * self.sensitivity) / self.smoothing
        step_y = (diff_y * self.sensitivity) / self.smoothing
        
        new_x = self.prev_x + step_x
        new_y = self.prev_y + step_y
        
        # Keep cursor on screen
        new_x = max(0, min(self.screen_width, new_x))
        new_y = max(0, min(self.screen_height, new_y))
        
        self.prev_x, self.prev_y = new_x, new_y
        return int(new_x), int(new_y)
    
    def show_gesture_guide(self):
        """Display gesture instructions"""
        guide = """
        🎮 GESTURE CONTROLS:
        
        👆 MOVE INDEX FINGER    = Move Cursor
        👌 PINCH & HOLD         = Left Click
        ✌️ PEACE SIGN & HOLD    = Right Click  
        🤜 FIST                 = Drag & Drop
        👍 THUMBS UP            = Open Browser
        
        Hold gestures for {:.1f} seconds to activate
        Press 'Q' in camera window to exit
        """.format(self.dwell_time)
        print(guide)
    
    def run(self):
        """Main application loop"""
        # User setup
        name, mobility_level = self.create_user_profile()
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not access camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Move to center initially
        pyautogui.moveTo(self.prev_x, self.prev_y)
        
        print("\n🚀 Starting gesture control...")
        self.show_gesture_guide()
        
        try:
            while self.tracking_active:
                success, frame = cap.read()
                if not success:
                    continue
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Get index finger position
                        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        
                        # Calculate cursor position
                        cursor_x, cursor_y = self.calculate_cursor_position(index_tip)
                        
                        # Move cursor (or drag if dragging)
                        if self.dragging:
                            pyautogui.dragTo(cursor_x, cursor_y, duration=0.01)
                        else:
                            pyautogui.moveTo(cursor_x, cursor_y)
                        
                        # Detect gestures
                        gesture = self.detect_gesture(hand_landmarks)
                        current_time = time.time()
                        
                        if gesture != "none":
                            # New gesture detected
                            if self.current_gesture != gesture:
                                self.current_gesture = gesture
                                self.gesture_start_time = current_time
                                print(f"👀 {gesture.replace('_', ' ').title()} detected...")
                            
                            # Gesture held long enough - execute action
                            elif current_time - self.gesture_start_time > self.dwell_time:
                                self.execute_action(gesture, (cursor_x, cursor_y))
                                self.current_gesture = "none"
                        else:
                            self.current_gesture = "none"
                
                # Display camera feed with info
                cv2.putText(frame, f"GestureAid - {name} ({mobility_level})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                status = "DRAGGING" if self.dragging else "READY"
                cv2.putText(frame, f"Status: {status}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                if self.current_gesture != "none":
                    hold_time = time.time() - self.gesture_start_time
                    progress = min(hold_time / self.dwell_time, 1.0)
                    cv2.putText(frame, 
                               f"Gesture: {self.current_gesture} ({progress:.1%})", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                cv2.imshow('GestureAid - Camera Feed (Press Q to quit)', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.dragging:
                pyautogui.mouseUp()
            print("🙏 Thank you for using GestureAid!")

# Run the application
if __name__ == "__main__":
    app = GestureAid()
    app.run()