import numpy as np
import cv2 as cv

def track_marker():
    cap = cv.VideoCapture(0)
    while not cap.isOpened():
        cap.open()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось прочитать кадр с камеры.")
            break
        marker_roi = detect_marker(frame)

        if marker_roi is not None:
            cv.imshow('Marker', marker_roi)
        if cv.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()

def detect_marker(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(largest_contour)
        marker_roi = frame[y:y+h, x:x+w]
        return marker_roi
    else:
        return None

if __name__ == "__main__":
    track_marker()





