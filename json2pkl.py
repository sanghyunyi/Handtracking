import json, glob, pickle

files_list = glob.glob('./data/train_*/*.json', recursive=True)

input_list = []
output_list = []
class_dic = {'pinching':0, 'clenching':1, 'poking':2, 'palming':3}

for file in files_list:
    with open(file) as jsonfile:
        action_class = file.split('/')[2][6:]
        output_list.append(class_dic[action_class])

        data = json.load(jsonfile)
        hand_coordinate = data['people'][0]['hand_right_keypoints_2d']
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

        input_list.append(normalized_hand_coordinate)

pickle.dump(input_list, open('./keypoint_list.pkl', 'wb'))
pickle.dump(output_list, open('./label_list.pkl', 'wb'))

