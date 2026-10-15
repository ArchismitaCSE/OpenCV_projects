
import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Webcam could not be opened.")
    exit()

while True:
    success, img = webcam.read()

    if not success:
        print("Error: Could not read webcam frame.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow('Face Detection', img)

    if cv2.waitKey(10) & 0xFF == 27:
        break

webcam.release()
cv2.destroyAllWindows()
