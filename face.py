import numpy as np
import cv2 as cv

def track_face():
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv.VideoCapture(0)
    while not cap.isOpened():
        cap.open()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось прочитать кадр с камеры.")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.imshow('Video', frame)
        if cv.waitKey(30) & 0xFF == 27:
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    track_face()
