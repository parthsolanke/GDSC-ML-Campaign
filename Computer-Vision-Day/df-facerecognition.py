import cv2
from deepface import DeepFace

# Load the pre-trained model
model = DeepFace.build_model("Gender")

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the webcam
    ret, frame = cap.read()

    # Perform gender and age detection
    result = DeepFace.analyze(frame, actions=['gender','age','emotion'], enforce_detection=False)
    # Extract gender, age, and emotion information
    emotion = result[0]['dominant_emotion']
    gender = result[0]['dominant_gender']
    age = result[0]['age']

    # Draw a rectangle around the detected face
    if 'region' in result[0]:
        region = result[0]['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the gender, age, and emotion data
    cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Gender: {gender}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Age: {age}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()