import cv2
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)

while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

        try:
            # Analyze face for emotion, age, and gender
            results = DeepFace.analyze(frame, actions=['emotion', 'age', 'gender'])[0]
            emotion = results['dominant_emotion']
            age = results['age']
            gender = results['gender']

            # Display results on the frame
            cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {age}", (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Gender: {gender}", (x, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        except:
            cv2.putText(frame, "No face detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


