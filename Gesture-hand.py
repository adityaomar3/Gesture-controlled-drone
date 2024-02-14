import cv2
import mediapipe as mp
import math

def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    angle_rad = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle_deg = math.degrees(angle_rad)
    return angle_deg + 360 if angle_deg < 0 else angle_deg

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def main():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

       
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
              
                    landmarks_list = []
                    for point in mp_hands.HandLandmark:
                        normalized_landmark = landmarks.landmark[point]
                        landmark_px = (int(normalized_landmark.x * frame.shape[1]),
                                       int(normalized_landmark.y * frame.shape[0]))
                        landmarks_list.append(landmark_px)

                    wrist, thumb, index = landmarks_list[0], landmarks_list[4], landmarks_list[8]
                    pitch_angle = calculate_angle(wrist, thumb, index)

                    print(f"Pitch Angle: {pitch_angle:.2f} degrees")
                    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('Hand Pose Estimation', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "_main_":
    main()