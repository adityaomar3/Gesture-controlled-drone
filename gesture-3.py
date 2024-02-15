import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Function to calculate angles
def calculate_angles(hand_landmarks, initial_wrist):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    
    # Calculate vectors
    thumb_vector = [thumb_tip.x - initial_wrist.x, thumb_tip.y - initial_wrist.y, thumb_tip.z - initial_wrist.z]
    index_vector = [index_tip.x - initial_wrist.x, index_tip.y - initial_wrist.y, index_tip.z - initial_wrist.z]
    middle_vector = [middle_tip.x - initial_wrist.x, middle_tip.y - initial_wrist.y, middle_tip.z - initial_wrist.z]

    # Calculate roll, pitch, and yaw angles
    roll_angle = math.degrees(math.atan2(thumb_vector[1], thumb_vector[0]))
    pitch_angle = math.degrees(math.atan2(thumb_vector[2], thumb_vector[0]))
    yaw_angle = math.degrees(math.atan2(index_vector[2], middle_vector[2]))

    return roll_angle, pitch_angle, yaw_angle

# Calibration angles
initial_roll, initial_pitch, initial_yaw = None, None, None

cap = cv2.VideoCapture(0)
drone_armed = False
# Define the bounding box
bbox = (100, 100, 1250, 1250)  # Format: (start_x, start_y, width, height)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        if initial_roll is None or initial_pitch is None or initial_yaw is None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]  # Assuming only one hand is detected
                initial_wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                initial_roll, initial_pitch, initial_yaw = calculate_angles(hand_landmarks, initial_wrist)

        # Draw the bounding box
        cv2.rectangle(image, (0,0), (150, 250), (0, 0,255), 2)

        # Crop the image to the bounding box
        #cropped_image = image[bbox[1]:bbox[2] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                roll, pitch, yaw = calculate_angles(hand_landmarks, initial_wrist)
                
                # Subtract initial angles for calibration
                calibrated_roll = roll - initial_roll
                calibrated_pitch = pitch - initial_pitch
                calibrated_yaw = yaw - initial_yaw
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
                index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                ring_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP] 
                thumb_middle_distance = ((thumb_tip.x - middle_pip.x)**2 +
                                    (thumb_tip.y - middle_pip.y)*2 )*0.5
                thumb_middlefingerpip_distance = ((thumb_tip.x - middle_tip.x)**2 +
                                    (thumb_tip.y - middle_tip.y)*2 )*0.5
                thumb_ring_tip_distance = ((thumb_tip.x - ring_tip.x)**2 +
                                    (thumb_tip.y - ring_tip.y)*2 )*0.5
                thumb_mcp_middle_tip_distance = ((thumb_mcp.x - middle_tip.x)**2 +
                                    (thumb_mcp.y - middle_tip.y)*2 )*0.5
                thumb_mcp_index_tip_distance = ((thumb_mcp.x - index_tip.x)**2 +
                                    (thumb_mcp.y - index_tip.y)*2 )*0.5
                thumb_tip_index_mcp_distance = ((thumb_tip.x - index_mcp.x)**2 +
                                    (thumb_tip.y - index_mcp.y)*2 )*0.5
                ring_dip_pinky_tip_distance = ((pinky_tip.x - ring_dip.x)**2 +
                                    (pinky_tip.y - ring_dip.y)*2 )*0.5
                thumb_tip_pinky_tip_distance = ((pinky_tip.x - thumb_tip.x)**2 +
                                    (pinky_tip.y - thumb_tip.y)*2 )*0.5
                middle_tip_index_tip_distance = ((index_tip.x - middle_tip.x)**2 +
                                    (index_tip.y - middle_tip.y)*2 )*0.5
                if thumb_middle_distance<0.05 and thumb_ring_tip_distance<0.05:
                    
                        print("Drone Armed!")
                        drone_armed = True
                elif thumb_middlefingerpip_distance<0.07 and thumb_mcp_middle_tip_distance<0.07:
                    
                        print("Drone Upwards!")
                elif thumb_tip_pinky_tip_distance<0.03 and middle_tip_index_tip_distance>0.09:
                     print("Drone Downwards!")
                elif thumb_mcp_index_tip_distance<0.08 and thumb_tip_index_mcp_distance>0.1:
                     print("Drone Left!")
                elif  thumb_tip_index_mcp_distance>0.1 and  ring_dip_pinky_tip_distance>0.1:
                     print("Drone Right!")
                else:
                     print("Drone Stop!")
                #print(f"Calibrated Roll: {calibrated_roll:.2f}, Calibrated Pitch: {calibrated_pitch:.2f}, Calibrated Yaw: {calibrated_yaw:.2f}")

        # Display the cropped image with hand landmarks
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(5) & 0xFF == 27:  
            break

cap.release()
cv2.destroyAllWindows()