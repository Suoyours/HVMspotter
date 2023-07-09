import cv2
import os
import numpy as np


image_dir = '/data/ys/PycharmProjects2/pan_pp.pytorch/data/container/container_v1/test/'
det_dir = '/data/ys/PycharmProjects2/pan_pp.pytorch/outputs/submit_swin_container_2classes/res_'

data_path = os.listdir(image_dir)
total_num = len(data_path)
num = 0
for image_name in data_path:
    print(str(num) + '/' + str(total_num) + '   ' + image_name)
    num += 1
    image_path = image_dir + image_name
    det_path = det_dir + image_name.split('.')[0] + '.txt'
    with open(det_path) as f:
        det_boxes = f.readlines()
        img = cv2.imread(image_path)
        det_boxes = [c.strip() for c in det_boxes]
        boxes = np.zeros((len(det_boxes), 4, 2))
        for i in range(len(det_boxes)):
            box_list = det_boxes[i].split(',')
            n = 0
            for j in range(4):
                for k in range(2):
                    boxes[i][j][k] = int(box_list[n])
                    n += 1
        for i, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            poly = poly.reshape(-1, 2)
            cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
        f.close()
        cv2.imwrite('/data/ys/PycharmProjects2/pan_pp.pytorch/outputs_images/test_swin100_2classes/res_' + image_name, img)
