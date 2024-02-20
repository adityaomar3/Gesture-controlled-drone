import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary to map predicted labels to human-readable strings
labels_dict = {0: 'Armed', 1: 'Disarm', 2: 'UP', 3: 'DOWN', 4: 'LEFT', 5: 'RIGHT', 6: 'FLIP'}

while True:
    # Initialize lists to store hand landmarks
    x_ = []
    y_ = []

    # Read frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    H, W, _ = frame.shape

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        # Draw hand landmarks and connections on the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Collect x and y coordinates of hand landmarks
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

        # Calculate bounding box coordinates
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Prepare data for prediction
        data_aux = []
        for i in range(len(x_)):
            x = x_[i]
            y = y_[i]
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        # Ensure data_aux has the correct number of features
        while len(data_aux) < 42:
            data_aux.extend([0, 0])

        # Make prediction
        prediction = model.predict([np.asarray(data_aux)])

        # Get predicted character
        predicted_character = labels_dict[int(prediction[0])]

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)

    # Check for ESC key to exit
    if cv2.waitKey(25) == 27:
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

