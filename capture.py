import cv2
import time


def get_images():

    camera = 0
    cap = cv2.VideoCapture(camera)

    # Capture X number of images
    for x in range(51):
        # take image
        ret, frame = cap.read()

        #   Mac default camera is 1280 by 720 (16:9)
        frame = frame[:, 160:1120, :]

        # display image
        cv2.imshow('frame', frame)
        print("Frame: " + str(x))

        # write image to file
        # Change name for different data
        if x != 0:
            out = cv2.imwrite('TrainingImages/two_fingers.' + str(x) + '.jpg', frame)
            cv2.waitKey(1)

            time.sleep(1)

    cap.release()


get_images()

'''
--thumbs_up
--open_hand
--one_finger
--closed_fist
two_fingers
'''
