import sys
import cv2
import os
from sys import platform
import argparse
import time
import itertools, pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import imagezmq

def keypt2input(hand_coordinate):
    hand_coordinate = list(itertools.chain.from_iterable(hand_coordinate))
    hand_coordinate_dic = {}

    x_list = []
    y_list = []
    c_list = []

    for i in range(21):
        x = hand_coordinate[3*i]
        y = hand_coordinate[3*i+1]
        c = hand_coordinate[3*i+2]

        if i == 0:
            x0 = x
            y0 = y

            x = x - x0
            y = y - y0
            c = c

        else:
            x = x - x0
            y = y - y0
            c = c

        hand_coordinate_dic[i] = [x, y, c]

        x_list.append(x)
        y_list.append(y)
        c_list.append(c)

    x_len = max(x_list) - min(x_list)
    y_len = max(y_list) - min(y_list)

    normalized_hand_coordinate = []

    for i in range(21):
        before_normalization = hand_coordinate_dic[i]
        norm_x = before_normalization[0]/x_len
        norm_y = before_normalization[1]/y_len
        norm_c = before_normalization[2]
        if i == 0:
            normalized_hand_coordinate += [norm_c]
        else:
            normalized_hand_coordinate += [norm_x, norm_y, norm_c]
        normalized_hand_coordinate += [norm_x**2 + norm_y**2]

    return normalized_hand_coordinate

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
def set_params():
    params = dict()
    params["model_folder"] = "../../../models/"
    params["hand"] = True
    params["hand_detector"] = 2
    params["body"] = 0
    params["write_json"] = "./data/"

    params["disable_multi_thread"] = True
    params["process_real_time"] = True
    params["output_resolution"] = "-1x-80"
    params["net_resolution"] = "-1x112"
    params["model_pose"] = "BODY_25" #It's faster on GPU?
    params["render_pose"] = 1
    params["number_people_max"] = 1

    #If GPU version is built, and multiple GPUs are available, set the ID here
    params["num_gpu_start"] = 0
    #params["disable_blending"] = False
    # Ensure you point to the correct path where models are located
    return params



clf = pickle.load(open('./pkls/trained_model.pkl','rb'))
scaler = pickle.load(open('./pkls/trained_scaler.pkl','rb'))
class_dic = {0:'pinching', 1:'clenching', 2:'poking', 3:'palming'}
imageHub = imagezmq.ImageHub()

params = set_params()
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
handRectangles = [
        [
        op.Rectangle(0.,0.,0.,0.), # Left hand
        op.Rectangle(42.,0.,108.,108.)  # Right hand
        ]
    ]
n = 0
while True:
    image_name, input_img = imageHub.recv_image()
    print(n)
    n += 1
    datum = op.Datum()
    datum.cvInputData = input_img
    datum.handRectangles = handRectangles
    opWrapper.emplaceAndPop([datum])

    righthandKeypoints = datum.handKeypoints[1][0]
    X = [keypt2input(righthandKeypoints)]
    X = scaler.transform(X)
    prob = clf.predict_proba(X)[0]
    print(prob)
    prob = prob[:-1] # drop palming
    reply = ",".join([str(p) for p in prob])
    imageHub.send_reply(str.encode(reply))

    # Display the stream
    cv2.imshow('Human Pose Estimation',datum.cvOutputData)
    cv2.waitKey(1)
