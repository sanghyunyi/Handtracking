# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import itertools, pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

clf = pickle.load(open('./pkls/trained_model.pkl','rb'))
scaler = pickle.load(open('./pkls/trained_scaler.pkl','rb'))
class_dic = {0:'pinching', 1:'clenching', 2:'poking', 3:'palming'}

def keypt2input(hand_coordinate):
    hand_coordinate = list(itertools.chain.from_iterable(hand_coordinate))
    hand_coordinate_dic = {}

    x_list = []
    y_list = []
    z_list = []

    for i in range(21):
        x = hand_coordinate[3*i]
        y = hand_coordinate[3*i+1]
        z = hand_coordinate[3*i+2]

        if i == 0:
            x0 = x
            y0 = y
            z0 = z
            x = x - x0
            y = y - y0
            z = z - z0

        else:
            x = x - x0
            y = y - y0
            z = z - z0

        hand_coordinate_dic[i] = [x, y, z]

        x_list.append(x)
        y_list.append(y)
        z_list.append(z)

    x_len = max(x_list) - min(x_list)
    y_len = max(y_list) - min(y_list)
    z_len = max(z_list) - min(z_list)

    normalized_hand_coordinate = []

    for i in range(21):
        before_normalization = hand_coordinate_dic[i]
        norm_x = before_normalization[0]/x_len
        norm_y = before_normalization[1]/y_len
        norm_z = before_normalization[2]/z_len
        normalized_hand_coordinate += [norm_x, norm_y, norm_z]
        normalized_hand_coordinate += [norm_x**2 + norm_y**2, norm_x**2 + norm_z**2, norm_y**2 + norm_z**2, norm_x**2 + norm_y**2 + norm_z**2]

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
    #params["write_json"] = "./data/"
    params["hand"] = True
    params["hand_detector"] = 2
    params["body"] = 0

    params["profile_speed"] = 10
    #params["hand_scale_number"] = 6
    #params["hand_scale_range"] = 0.4
    params["disable_multi_thread"] = True
    params["process_real_time"] = True
    params["output_resolution"] = "-1x-80"
    params["net_resolution"] = "-1x144"
    params["model_pose"] = "BODY_25" #It's faster on GPU?
    #params["model_pose"] = "COCO"
    params["number_people_max"] = 1
    #params["alpha_pose"] = 0.6
    #params["scale_gap"] = 0.3
    #params["scale_number"] = 1
    #params["render_threshold"] = 0.05
    #params["logging_level"] = 3
    #params["write_video"] = "./test.avi"

    #If GPU version is built, and multiple GPUs are available, set the ID here
    #params["num_gpu_start"] = 0
    #params["disable_blending"] = False
    # Ensure you point to the correct path where models are located
    return params


# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()

def main():
    params = set_params()

    #Constructing OpenPose object allocates GPU memory
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    #Opening OpenCV stream
    stream = cv2.VideoCapture(1)
    stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #video_out = cv2.VideoWriter('output.avi', fourcc, 10, (int(stream.get(3)), int(stream.get(4))))

    handRectangles = [
            [
            op.Rectangle(0.,0.,0.,0.), # Left hand
            op.Rectangle(0.,0.,1280.,1280.)  # Right hand
            ]
        ]
    n = 0
    while True:
        ret,img = stream.read()

        # Display the stream
        cv2.imshow('Human Pose Estimation',img)

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
