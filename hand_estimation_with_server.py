# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
import imagezmq
import time

url = 'tcp://54.188.75.251:5555'#/actionclass'

sender = imagezmq.ImageSender(connect_to=url)

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def main():
    #Opening OpenCV stream
    stream = cv2.VideoCapture(1)
    stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #video_out = cv2.VideoWriter('output.avi', fourcc, 10, (int(stream.get(3)), int(stream.get(4))))

    n = 0
    while True:
        ret, img = stream.read()
        #img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB) # Change rgb video to monochrome

        img = rescale_frame(img, percent=10)
        #video_out.write(img)
        start = time.time()
        prediction = sender.send_image('HandPose', img)

        print(time.time() - start)

        print(prediction.decode())

        # Display the stream
        #cv2.imshow('Human Pose Estimation', img)

        #key = cv2.waitKey(0)
        key = cv2.waitKey(1)

        n += 1
        if key==ord('q'):
            break

    stream.release()
    #video_out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
        main()
