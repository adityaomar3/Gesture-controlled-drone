import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Video capture
cap = cv2.VideoCapture(0)

# Flag for drone arm/disarm
drone_armed = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmark coordinates
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Calculate distance between thumb and index finger tips
            thumb_index_distance = ((thumb_tip.x - index_tip.x)**2 +(thumb_tip.y - index_tip.y)**2 + (thumb_tip.z-index_tip.z)**2)**0.5

            # Arm the drone if thumb and index fingers are close (fist)
            if thumb_index_distance < 0.03:
                if not drone_armed:
                    print("Drone Armed!")
                    drone_armed = True

            # Disarm the drone if fingers are apart (open hand)
            else:
                if drone_armed:
                    print("Drone Disarmed!")
                    drone_armed = False

    # Display the frame
    cv2.imshow("Drone Control", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()