import cv2
import time

def get_images():
    print("Get ready...")
    time.sleep(3)

    camera = 0
    cap = cv2.VideoCapture(camera)
    cap.set(3,960)
    cap.set(4,720)

    ret, frame = cap.read()

    cv2.imshow('frame', frame)
    out = cv2.imwrite('PredictData/1.jpg', frame)
    cv2.waitKey(1)

    time.sleep(1)

    cap.release()
get_images()
