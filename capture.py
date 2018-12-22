import cv2
import time


def get_images(ofWhat):

    camera = 0
    cap = cv2.VideoCapture(camera)
    cap.set(3,960)

    cap.set(4,720)

    # Capture X number of images
    for x in range(51,100):
        # take image
        ret, frame = cap.read()

        #   Mac default camera is 1280 by 720 (16:9)
        # frame = frame[:, 160:1120, :]

        # display image
        cv2.imshow('frame', frame)
        print("Frame: " + str(x))

        # write image to file
        # Change name for different data

        out = cv2.imwrite('TrainingImages/' + ofWhat + "." + str(x) + '.jpg', frame)
        cv2.waitKey(1)

        time.sleep(1)

    cap.release()


for i in ["thumbs_up","open_hand","one_finger","closed_fist","two_finger"]:
    print(i)
    time.sleep(5)
    get_images(i)
    time.sleep(2)


'''
--thumbs_up
--open_hand
--one_finger
--closed_fist
two_fingers
'''
