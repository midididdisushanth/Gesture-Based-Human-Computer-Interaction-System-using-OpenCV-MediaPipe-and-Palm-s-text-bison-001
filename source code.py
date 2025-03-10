import cv2
import mediapipe as mp
import time
import os

# Initialize MediaPipe Hand Tracking with improved parameters
mpHands = mp.solutions.hands
# Setting higher min_detection_confidence and min_tracking_confidence for more stability
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Focus on one hand for better performance
    min_detection_confidence=0.7,  # Higher confidence threshold
    min_tracking_confidence=0.5  # Better tracking between frames
)
mpDraw = mp.solutions.drawing_utils

# Capture Video
cap = cv2.VideoCapture(0)
# Set lower resolution for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Gesture Definitions (Mapped to NirCmd Commands)
GESTURE_ACTIONS = {
    "thumbs_up": "volume_up",
    "thumbs_down": "volume_down",
    "fist": "mute_audio",
    "open_palm": "play_pause",
    }

# Cooldown mechanism to prevent rapid-fire commands
last_command_time = 0
COMMAND_COOLDOWN = 1.0  # seconds between commands

# Add gesture confidence tracking
gesture_history = []
HISTORY_LENGTH = 5  # Number of frames to consider
CONFIDENCE_THRESHOLD = 0.6  # Percentage of consistent readings needed

def execute_command(command):
    """Executes system-level commands based on recognized gestures."""
    if command == "volume_up":
        os.system('nircmd.exe changesysvolume 15000')  # Increase volume
    elif command == "volume_down":
        os.system('nircmd.exe changesysvolume -15000')  # Decrease volume
    elif command == "mute_audio":
        os.system('nircmd.exe mutesysvolume 2')  # Toggle mute
    elif command == "play_pause":
        os.system('nircmd.exe sendkeypress space')  # Simulates spacebar (for Play/Pause)
    elif command == "next_track":
        os.system('nircmd.exe sendkeypress media_next')  # Next track
    elif command == "previous_track":
        os.system('nircmd.exe sendkeypress media_prev')  # Previous track
    else:
        print("Unknown command received:", command)

def detect_gesture(handLms):
    """Improved gesture detection logic with more robust landmarks analysis"""
    # Get key landmarks for gesture recognition
    thumb_tip = handLms.landmark[4]
    index_tip = handLms.landmark[8]
    middle_tip = handLms.landmark[12]
    ring_tip = handLms.landmark[16]
    pinky_tip = handLms.landmark[20]
    
    # Get wrist position for reference
    wrist = handLms.landmark[0]
    
    # Check if fingers are extended by comparing with middle knuckles
    thumb_extended = thumb_tip.x > handLms.landmark[3].x if wrist.x < thumb_tip.x else thumb_tip.x < handLms.landmark[3].x
    index_extended = index_tip.y < handLms.landmark[6].y
    middle_extended = middle_tip.y < handLms.landmark[10].y
    ring_extended = ring_tip.y < handLms.landmark[14].y
    pinky_extended = pinky_tip.y < handLms.landmark[18].y
    
    # Count extended fingers
    extended_fingers = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
    finger_count = sum(extended_fingers)
    
    # Detect thumbs up/down with more precise positioning
    if thumb_extended and not any(extended_fingers[1:]):  # Only thumb is up
        # Check thumb direction (up vs down)
        if thumb_tip.y < wrist.y:  # Thumb is above wrist
            return "thumbs_up"
        else:  # Thumb is below wrist
            return "thumbs_down"
    
    # Detect fist - no fingers extended
    elif finger_count == 0:
        return "fist"
    
    # Detect open palm - all fingers extended
    elif finger_count == 5:
        return "open_palm"
    
    # Could add swipe detection by tracking hand movement over time
    
    return None  # No recognized gesture

# Main Loop
prev_hand_center_x = None
frame_count = 0

while True:
    success, img = cap.read()
    if not success:
        continue
        
    # Process every other frame for better performance
    frame_count += 1
    if frame_count % 2 != 0:
        cv2.imshow("Hand Gesture Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # Start timing for performance monitoring
    start_time = time.time()
    
    # Convert to RGB for MediaPipe
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    detected_gesture = None
    current_time = time.time()
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
            # Improved gesture detection
            detected_gesture = detect_gesture(handLms)
            
            # Track hand position for potential swipe detection
            # hand_center_x = handLms.landmark[9].x
            # if prev_hand_center_x is not None:
            #     # Detect horizontal swipes
            #     if hand_center_x - prev_hand_center_x > 0.05:  # Moving right
            #         detected_gesture = "swipe_right"
            #     elif prev_hand_center_x - hand_center_x > 0.05:  # Moving left
            #         detected_gesture = "swipe_left"
            # prev_hand_center_x = hand_center_x
            
            # Add to gesture history for confidence tracking
            gesture_history.append(detected_gesture)
            if len(gesture_history) > HISTORY_LENGTH:
                gesture_history.pop(0)
    else:
        gesture_history.append(None)
        if len(gesture_history) > HISTORY_LENGTH:
            gesture_history.pop(0)
        prev_hand_center_x = None
    
    # Only process gesture if we have a consistent reading
    if len(gesture_history) == HISTORY_LENGTH:
        most_common = max(set(filter(None, gesture_history)), key=gesture_history.count, default=None)
        confidence = gesture_history.count(most_common) / HISTORY_LENGTH if most_common else 0
        
        # Execute command if confidence is high enough and cooldown period has passed
        if most_common and confidence > CONFIDENCE_THRESHOLD and most_common in GESTURE_ACTIONS:
            if current_time - last_command_time > COMMAND_COOLDOWN:
                print(f"Recognized Gesture: {most_common} (Confidence: {confidence:.2f})")
                execute_command(GESTURE_ACTIONS[most_common])
                last_command_time = current_time
    
    # Calculate and display FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show image with any detected gesture
    if detected_gesture:
        cv2.putText(img, f"Gesture: {detected_gesture}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Hand Gesture Control", img)
    
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
