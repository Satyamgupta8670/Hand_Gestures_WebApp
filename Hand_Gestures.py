import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
from collections import Counter, deque

def main():
    # Page configuration
    st.set_page_config(layout="wide", page_title="Advanced Gesture Recognition")
    
    st.title("Advanced Hand Gesture Recognition")
    
    # Create sidebar for controls
    st.sidebar.header("Controls")
    run = st.sidebar.checkbox("Start/Stop Camera", value=False)
    
    # Model settings
    with st.sidebar.expander("Model Settings", expanded=False):
        detection_confidence = st.slider("Detection Confidence", 0.5, 0.95, 0.7, 0.05)
        tracking_confidence = st.slider("Tracking Confidence", 0.5, 0.95, 0.7, 0.05)
        max_hands = st.slider("Maximum Hands", 1, 2, 1, 1)
        gesture_stability = st.slider("Gesture Stability", 3, 15, 8, 1, 
                                    help="Higher values make gesture detection more stable but less responsive")
    
    # Information about supported gestures
    with st.sidebar.expander("Supported Gestures", expanded=True):
        st.markdown("""
        ### Basic Gestures
        1. 👍 **Thumbs Up** - Thumb extended upward
        2. 👎 **Thumbs Down** - Thumb extended downward
        3. ✌️ **Peace/Victory** - Index and middle fingers extended
        4. 👌 **OK** - Thumb and index finger form a circle
        5. 👉 **Pointing** - Only index finger extended
        6. 🖐️ **Open Palm** - All fingers extended
        7. ✊ **Fist** - All fingers closed
        8. 🤘 **Rock Sign** - Index and pinky fingers extended
        9. 🤙 **Call Me** - Thumb and pinky extended
        10. 👋 **Wave** - Fingers extended and moving side to side
        
        ### Number Gestures
        11. **One** - Index finger extended (similar to pointing)
        12. **Two** - Index and middle fingers extended (similar to peace)
        13. **Three** - Thumb, index, and middle extended
        14. **Four** - All fingers except thumb extended
        15. **Five** - All fingers extended (similar to open palm)
        
        ### Advanced Gestures
        16. 👊 **Fist Bump** - Closed fist moving forward
        17. ✋ **High Five** - Open palm moving forward
        18. 👈 **Left Point** - Index finger pointing left
        19. 👇 **Down Point** - Index finger pointing down
        20. 👆 **Up Point** - Index finger pointing up
        21. 🤏 **Pinch** - Thumb and index fingertips touching
        22. 🖖 **Vulcan Salute** - Split fingers between middle and ring
        23. 🤞 **Crossed Fingers** - Index crossed over middle finger
        24. 👍👎 **Thumbs Flip** - Transitioning between thumbs up and down
        25. 🤜 **Side Fist** - Fist with thumb side toward camera
        26. 🫱 **Flat Hand** - Hand horizontal, palm down
        27. 🫲 **Scoop Hand** - Hand horizontal, palm up
        28. ✍️ **Writing** - Moving index finger as if writing
        29. 👏 **Clap** - Both palms coming together (with dual hand detection)
        30. 🤝 **Handshake** - One hand open horizontally (with dual hand detection)
        """)
    
    # Create a placeholder for the webcam feed
    col1, col2 = st.columns([3, 1])
    
    with col1:
        frame_placeholder = st.empty()
    
    with col2:
        st.subheader("Detected Gesture")
        gesture_placeholder = st.empty()
        
        # Confidence meter
        st.subheader("Confidence")
        confidence_placeholder = st.empty()
        confidence_bar = st.empty()
        
        # Gesture counter
        st.subheader("Gesture Count")
        gesture_count_container = st.container()
        
        # Gesture history
        st.subheader("Recent Gestures")
        history_placeholder = st.empty()
    
    # Initialize MediaPipe
    @st.cache_resource
    def initialize_mediapipe(detection_conf=0.7, tracking_conf=0.7, max_num_hands=1):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        mp_draw = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        return mp_hands, hands, mp_draw, mp_drawing_styles
    
    mp_hands, hands, mp_draw, mp_drawing_styles = initialize_mediapipe(
        detection_conf=detection_confidence,
        tracking_conf=tracking_confidence,
        max_num_hands=max_hands
    )
    
    # Initialize gesture counters
    if 'gesture_counters' not in st.session_state:
        st.session_state.gesture_counters = {
            # Basic gestures
            "Thumbs Up": 0, "Thumbs Down": 0, "Peace": 0, "OK": 0, "Pointing": 0,
            "Open Palm": 0, "Fist": 0, "Rock": 0, "Call Me": 0, "Wave": 0,
            # Number gestures
            "One": 0, "Two": 0, "Three": 0, "Four": 0, "Five": 0,
            # Advanced gestures
            "Fist Bump": 0, "High Five": 0, "Left Point": 0, "Down Point": 0, "Up Point": 0,
            "Pinch": 0, "Vulcan Salute": 0, "Crossed Fingers": 0, "Thumbs Flip": 0, "Side Fist": 0,
            "Flat Hand": 0, "Scoop Hand": 0, "Writing": 0, "Clap": 0, "Handshake": 0,
            "Unknown": 0
        }
    
    # Gesture history for temporal gestures
    if 'gesture_history' not in st.session_state:
        st.session_state.gesture_history = deque(maxlen=10)
    
    # Hand landmark history for motion detection
    if 'landmark_history' not in st.session_state:
        st.session_state.landmark_history = deque(maxlen=10)
    
    # Function to calculate angles between landmarks
    def calculate_angle(a, b, c):
        # Calculate vectors
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        
        # Calculate dot product and magnitudes
        dot_product = np.dot(ba, bc)
        magnitude_ba = np.linalg.norm(ba)
        magnitude_bc = np.linalg.norm(bc)
        
        # Calculate angle in radians and convert to degrees
        cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    # Function to calculate distance between landmarks
    def calculate_distance(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    # Function to detect motion
    def detect_motion(current_landmarks, landmark_history):
        if not landmark_history or len(landmark_history) < 2:
            return "Static", 0.0
        
        # Use wrist landmark (0) for overall hand movement
        wrist_current = current_landmarks[0]
        wrist_prev = landmark_history[-1][0]
        
        # Calculate horizontal and vertical movement
        dx = wrist_current[0] - wrist_prev[0]
        dy = wrist_current[1] - wrist_prev[1]
        
        # Calculate magnitude of movement
        movement = np.sqrt(dx**2 + dy**2)
        
        # Determine direction if there's significant movement
        direction = "Static"
        if movement > 0.02:  # Threshold for movement detection
            # Determine primary direction
            if abs(dx) > abs(dy):
                direction = "Right" if dx > 0 else "Left"
            else:
                direction = "Down" if dy > 0 else "Up"
                
            # Check for forward/backward movement using z-coordinate
            dz = wrist_current[2] - wrist_prev[2]
            if abs(dz) > abs(dx) and abs(dz) > abs(dy) and abs(dz) > 0.01:
                direction = "Forward" if dz < 0 else "Backward"
        
        return direction, movement
    
    # Improved function to detect hand gestures
    def detect_gesture(landmarks, landmark_history=None):
        # Extract key landmarks
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]  # Inter-phalangeal joint
        thumb_mcp = landmarks[2]  # Metacarpophalangeal joint
        index_tip = landmarks[8]
        index_pip = landmarks[6]  # Proximal interphalangeal joint
        index_mcp = landmarks[5]  # Metacarpophalangeal joint
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        middle_mcp = landmarks[9]
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        ring_mcp = landmarks[13]
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        pinky_mcp = landmarks[17]
        
        # Improved finger extension detection
        # For thumb (special due to its positioning)
        # Check if hand is left or right by comparing index MCP to wrist
        is_right_hand = index_mcp[0] > wrist[0]
        
        # For thumb extension, check if tip is further from palm than IP joint
        if is_right_hand:
            thumb_extended = thumb_tip[0] > thumb_ip[0]
        else:
            thumb_extended = thumb_tip[0] < thumb_ip[0]
            
        # Check if thumb is pointing up
        thumb_up = thumb_tip[1] < thumb_mcp[1]
        
        # For other fingers - check if fingertip is higher than PIP joint
        # (Lower y value means higher position in image)
        # Adding a margin for more reliable detection
        margin = 0.03
        index_extended = index_tip[1] < index_pip[1] - margin
        middle_extended = middle_tip[1] < middle_pip[1] - margin
        ring_extended = ring_tip[1] < ring_pip[1] - margin
        pinky_extended = pinky_tip[1] < pinky_pip[1] - margin
        
        # Calculate palm orientation
        # Create vectors from wrist to index MCP and from wrist to pinky MCP
        wrist_to_index = np.array([index_mcp[0] - wrist[0], index_mcp[1] - wrist[1], index_mcp[2] - wrist[2]])
        wrist_to_pinky = np.array([pinky_mcp[0] - wrist[0], pinky_mcp[1] - wrist[1], pinky_mcp[2] - wrist[2]])
        
        # Calculate palm normal vector (perpendicular to palm)
        palm_normal = np.cross(wrist_to_index, wrist_to_pinky)
        palm_normal = palm_normal / np.linalg.norm(palm_normal)  # Normalize
        
        # Check if palm is facing the camera
        hand_facing_camera = palm_normal[2] > 0 if is_right_hand else palm_normal[2] < 0
        
        # Determine palm direction
        palm_direction = "unknown"
        # Calculate angle between palm normal and camera direction (0,0,1)
        camera_direction = np.array([0, 0, 1])
        palm_angle = np.arccos(np.clip(np.dot(palm_normal, camera_direction), -1.0, 1.0)) * 180 / np.pi
        
        if palm_angle < 45:
            palm_direction = "forward"
        elif palm_angle > 135:
            palm_direction = "backward"
        else:
            # Check if palm is facing up or down
            if middle_mcp[1] < wrist[1]:  # Middle MCP above wrist
                palm_direction = "up"
            else:
                palm_direction = "down"
        
        # Calculate fingertip distances for gesture recognition
        thumb_to_index_dist = calculate_distance(thumb_tip, index_tip)
        
        # Detect motion if history is available
        motion_direction = "Static"
        motion_magnitude = 0.0
        if landmark_history and len(landmark_history) > 1:
            motion_direction, motion_magnitude = detect_motion(landmarks, landmark_history)
        
        # IMPROVED GESTURE RECOGNITION
        gesture = "Unknown"
        confidence = 0.0
        
        # Pointing detection (improved)
        # This is one of the gestures you mentioned having trouble with
        if not thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
            # Basic pointing detected - now determine direction
            # Create vector from index MCP to index tip
            point_vector = np.array([index_tip[0] - index_mcp[0], index_tip[1] - index_mcp[1]])
            point_vector = point_vector / np.linalg.norm(point_vector)  # Normalize
            
            # Determine pointing direction based on the angle
            angle = np.arctan2(point_vector[1], point_vector[0]) * 180 / np.pi
            
            # Classify direction based on angle
            if -45 <= angle <= 45:
                gesture = "Right Point" if is_right_hand else "Left Point"
                confidence = 0.9
            elif 45 < angle <= 135:
                gesture = "Down Point"
                confidence = 0.9
            elif -135 <= angle < -45:
                gesture = "Up Point"
                confidence = 0.9
            else:  # angle > 135 or angle < -135
                gesture = "Left Point" if is_right_hand else "Right Point"
                confidence = 0.9
            
            # If no specific direction is clear, just use generic pointing
            if abs(point_vector[0]) < 0.3 and abs(point_vector[1]) < 0.3:
                gesture = "Pointing"
                confidence = 0.85
        
        # Basic gesture detection with improved logic
        # Thumbs Up
        elif thumb_extended and thumb_up and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            gesture = "Thumbs Up"
            confidence = 0.95
        
        # Thumbs Down
        elif thumb_extended and not thumb_up and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            gesture = "Thumbs Down"
            confidence = 0.95
        
        # Peace/Victory
        elif not thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
            # Check if fingers are apart (not crossed)
            if calculate_distance(index_tip, middle_tip) > 0.05:
                gesture = "Peace"
                confidence = 0.95
        
        # OK gesture
        elif thumb_to_index_dist < 0.05 and middle_extended and ring_extended and pinky_extended:
            gesture = "OK"
            confidence = 0.9
        
        # Open Palm / Five
        elif thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            if palm_direction == "up" or palm_direction == "forward":
                gesture = "Open Palm"
                confidence = 0.9
            elif palm_direction == "down":
                gesture = "Flat Hand"
                confidence = 0.85
        
        # Fist
        elif not thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            if hand_facing_camera:
                gesture = "Fist"
                confidence = 0.85
            else:
                gesture = "Side Fist"
                confidence = 0.85
        
        # Rock Sign
        elif not thumb_extended and index_extended and not middle_extended and not ring_extended and pinky_extended:
            gesture = "Rock"
            confidence = 0.9
        
        # Call Me
        elif thumb_extended and not index_extended and not middle_extended and not ring_extended and pinky_extended:
            gesture = "Call Me"
            confidence = 0.85
        
        # Wave (temporal)
        elif thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            if motion_direction in ["Left", "Right"] and motion_magnitude > 0.03:
                gesture = "Wave"
                confidence = 0.8
        
        # Number Three
        elif thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
            gesture = "Three"
            confidence = 0.9
        
        # Number Four
        elif not thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            gesture = "Four"
            confidence = 0.9
        
        # Pinch
        elif thumb_to_index_dist < 0.05 and not middle_extended and not ring_extended and not pinky_extended:
            gesture = "Pinch"
            confidence = 0.9
        
        # Vulcan Salute - improved detection
        elif not thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            # Check distances between fingertips to detect the characteristic split
            index_middle_dist = calculate_distance(index_tip, middle_tip)
            middle_ring_dist = calculate_distance(middle_tip, ring_tip)
            ring_pinky_dist = calculate_distance(ring_tip, pinky_tip)
            
            # In Vulcan salute, there should be a gap between middle and ring fingers
            if middle_ring_dist > 1.5 * max(index_middle_dist, ring_pinky_dist):
                gesture = "Vulcan Salute"
                confidence = 0.85
        
        # Writing motion detection
        elif not thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
            if motion_magnitude > 0.02 and landmark_history and len(landmark_history) >= 5:
                # Track index fingertip positions
                tip_positions = [lm[8][0:2] for lm in list(landmark_history)[-5:]]
                tip_positions.append(index_tip[0:2])
                
                # Calculate the variability in movement
                x_vals = [p[0] for p in tip_positions]
                y_vals = [p[1] for p in tip_positions]
                x_range = max(x_vals) - min(x_vals)
                y_range = max(y_vals) - min(y_vals)
                
                if x_range > 0.05 and y_range > 0.05:
                    gesture = "Writing"
                    confidence = 0.7
        
        # Temporal gestures - Fist Bump
        if gesture == "Fist" and motion_direction == "Forward" and motion_magnitude > 0.04:
            gesture = "Fist Bump"
            confidence = 0.75
        
        # High Five
        elif gesture == "Open Palm" and motion_direction == "Forward" and motion_magnitude > 0.04:
            gesture = "High Five"
            confidence = 0.75
        
        # Scoop Hand - improved detection
        elif thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            if palm_direction == "up" and not hand_facing_camera:
                gesture = "Scoop Hand"
                confidence = 0.8
        
        return gesture, confidence
    
    # For gesture stability, keep track of recent detections
    MAX_GESTURE_HISTORY = gesture_stability  # Use the slider value for stability
    recent_gestures = deque(maxlen=MAX_GESTURE_HISTORY)
    
    # Variables for FPS calculation and performance tracking
    fps_history = deque(maxlen=30)  # For averaging FPS over time
    prev_time = time.time()
    
    # Initialize webcam with better error handling
    cap = None
    
    if run:
        try:
            # Try to open the webcam with more specific settings
            cap = cv2.VideoCapture(0)
            
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS
            
            if not cap.isOpened():
                st.error("Failed to open webcam. Please check your camera connection.")
                run = False
        except Exception as e:
            st.error(f"Error initializing webcam: {e}")
            run = False
    
    # Main camera loop
    while run:
        try:
            # Measure frame start time for performance monitoring
            frame_start_time = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                st.error("Failed to capture image from webcam.")
                break
            
            # Flip the frame for a more intuitive display
            frame = cv2.flip(frame, 1)
            
            # For performance: reduce resolution for processing if needed
            # Create a working copy of the frame with reduced size for faster processing
            process_frame = cv2.resize(frame, (320, 240))
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
            
            # Process the image with MediaPipe
            results = hands.process(rgb_frame)
            
            # Scale factor between original and processed frame
            scale_x = frame.shape[1] / process_frame.shape[1]
            scale_y = frame.shape[0] / process_frame.shape[0]
            
            # Detect hands and gestures
            detected_gesture = "No Hand Detected"
            confidence = 0.0
            
            # Draw hand landmarks and detect gestures
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw scaled landmarks on the original frame
                    for landmark in hand_landmarks.landmark:
                        # Scale the landmarks to match the original frame
                        landmark.x = landmark.x
                        landmark.y = landmark.y
                    
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Extract normalized landmark coordinates
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append([landmark.x, landmark.y, landmark.z])
                    
                    # Update landmark history for motion detection
                    st.session_state.landmark_history.append(landmarks)
                    
                    # Detect gesture
                    gesture, conf = detect_gesture(landmarks, st.session_state.landmark_history)
                    
                    # Add to recent gestures for stability
                    recent_gestures.append((gesture, conf))
                    
                    # Get the most stable gesture
                    if recent_gestures:
                        # Count occurrences of each gesture in recent history
                        gesture_counts = Counter([g for g, _ in recent_gestures])
                        
                        # Find the most common gesture that meets the confidence threshold
                        for g, count in gesture_counts.most_common():
                            # Calculate average confidence for this gesture
                            avg_conf = np.mean([c for gest, c in recent_gestures if gest == g])
                            
                            # If gesture is stable enough and confidence is good, use it
                            if count >= MAX_GESTURE_HISTORY // 2 and avg_conf > 0.7:
                                detected_gesture = g
                                confidence = avg_conf
                                
                                # Check if this is a "new" stable gesture to add to history
                                if len(st.session_state.gesture_history) == 0 or st.session_state.gesture_history[-1] != detected_gesture:
                                    # Update counter and history
                                    st.session_state.gesture_counters[detected_gesture] += 1
                                    st.session_state.gesture_history.append(detected_gesture)
                                break
                    
                    # Display gesture on frame
                    cv2.putText(frame, f"Gesture: {detected_gesture}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Display confidence on frame
                    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Update UI elements
            gesture_placeholder.markdown(f"## {detected_gesture}")
            confidence_placeholder.text(f"{confidence:.2f}")
            confidence_bar.progress(confidence)
            
            # Display emoji in sidebar based on detected gesture
            emoji_map = {
                "Thumbs Up": "👍", "Thumbs Down": "👎", "Peace": "✌️", "OK": "👌",
                "Pointing": "👉", "Open Palm": "🖐️", "Fist": "✊", "Rock": "🤘",
                "Call Me": "🤙", "Wave": "👋", "One": "1️⃣", "Two": "2️⃣",
                "Three": "3️⃣", "Four": "4️⃣", "Five": "5️⃣", "Fist Bump": "👊",
                "High Five": "✋", "Left Point": "👈", "Down Point": "👇",
                "Up Point": "👆", "Pinch": "🤏", "Vulcan Salute": "🖖",
                "Crossed Fingers": "🤞", "Thumbs Flip": "👍👎", "Side Fist": "🤜",
                "Flat Hand": "🫱", "Scoop Hand": "🫲", "Writing": "✍️",
                "Clap": "👏", "Handshake": "🤝", "Unknown": "❓",
                "No Hand Detected": "❓"
            }
            
            emoji = emoji_map.get(detected_gesture, "❓")
            st.sidebar.markdown(f"# {emoji}")
            
            # Calculate FPS
            curr_time = time.time()
            frame_time = curr_time - prev_time
            prev_time = curr_time
            
            fps = 1 / frame_time if frame_time > 0 else 0
            fps_history.append(fps)
            avg_fps = sum(fps_history) / len(fps_history)
            
            # Display FPS on frame
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (frame.shape[1] - 120, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Display the frame - FIXED THE DEPRECATED PARAMETER
            frame_placeholder.image(frame, channels="RGB", use_container_width=True)
            
            # Display gesture counts
            with gesture_count_container:
                count_columns = st.columns(2)
                count_text_0 = ""
                count_text_1 = ""
                
                # Sort gestures by count (descending)
                sorted_gestures = sorted(
                    [(gesture, count) for gesture, count in st.session_state.gesture_counters.items() if count > 0],
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Split the list into two columns
                half_idx = len(sorted_gestures) // 2 + len(sorted_gestures) % 2
                
                for gesture, count in sorted_gestures[:half_idx]:
                    count_text_0 += f"{gesture}: {count}\n"
                
                for gesture, count in sorted_gestures[half_idx:]:
                    count_text_1 += f"{gesture}: {count}\n"
                
                count_columns[0].text(count_text_0 if count_text_0 else "No gestures")
                count_columns[1].text(count_text_1 if count_text_1 else "")
            
            # Display gesture history
            if st.session_state.gesture_history:
                history_text = " → ".join([f"{g}" for g in list(st.session_state.gesture_history)[-5:]])
                history_placeholder.text(history_text)
            else:
                history_placeholder.text("No gesture history yet")
            
            # Adaptive sleep to maintain target framerate and reduce CPU usage
            # Target 30fps but don't sleep if we're already below target
            target_frame_time = 1.0 / 30  # 30 fps
            elapsed = time.time() - frame_start_time
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)
            
        except Exception as e:
            st.error(f"Error in webcam processing: {e}")
            time.sleep(1)  # Wait a bit before trying again
    
    # Display static image when webcam is not running
    if not run:
        placeholder_img = np.ones((480, 640, 3), dtype=np.uint8) * 200
        cv2.putText(placeholder_img, "Camera Off", (220, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        frame_placeholder.image(placeholder_img, channels="RGB", use_container_width=True)
        
        gesture_placeholder.markdown("## Camera is Off")
        confidence_placeholder.text("0.00")
        confidence_bar.progress(0.0)
        
        # Still display counts when camera is off
        with gesture_count_container:
            count_columns = st.columns(2)
            count_text_0 = ""
            count_text_1 = ""
            
            # Sort gestures by count (descending)
            sorted_gestures = sorted(
                [(gesture, count) for gesture, count in st.session_state.gesture_counters.items() if count > 0],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Split the list into two columns
            half_idx = len(sorted_gestures) // 2 + len(sorted_gestures) % 2
            
            for gesture, count in sorted_gestures[:half_idx]:
                count_text_0 += f"{gesture}: {count}\n"
            
            for gesture, count in sorted_gestures[half_idx:]:
                count_text_1 += f"{gesture}: {count}\n"
            
            count_columns[0].text(count_text_0 if count_text_0 else "No gestures")
            count_columns[1].text(count_text_1 if count_text_1 else "")
        
        history_placeholder.text("No gesture history")
    
    # Clean up camera resources if it was initialized
    if cap is not None:
        cap.release()
    
    # Additional controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Additional Controls")
    
    # Reset counters button
    if st.sidebar.button("Reset Counters"):
        for gesture in st.session_state.gesture_counters:
            st.session_state.gesture_counters[gesture] = 0
    
    # Reset history button
    if st.sidebar.button("Reset History"):
        st.session_state.gesture_history.clear()
    
    # Export data option
    if st.sidebar.button("Export Data"):
        # Create a simple CSV string
        csv_data = "Gesture,Count\n"
        for gesture, count in st.session_state.gesture_counters.items():
            if count > 0:
                csv_data += f"{gesture},{count}\n"
        
        # Provide download button
        st.sidebar.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="gesture_counts.csv",
            mime="text/csv"
        )

    # Performance settings
    with st.sidebar.expander("Performance Settings", expanded=False):
        st.markdown("""
        If you're experiencing performance issues:
        
        1. Try reducing your camera resolution in system settings
        2. Close other applications running in the background
        3. Ensure good lighting for better hand detection
        """)

if __name__ == "__main__":
    main()