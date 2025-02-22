import os
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import cv2
import shutil

image_format = '.jpg'
image_preprocessing_output = 'preprocessing'

w = 640
h = 360

# delete all previous files
def clear_preprocessing():
    for filename in os.listdir(image_preprocessing_output):
        file_path = os.path.join(image_preprocessing_output, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def image_preprocessing(input_path: str) -> int:
    print(input_path)
    clear_preprocessing()
    entries = os.listdir(input_path+'/')
    entries = [x for x in entries if x.endswith(image_format)]
    length = len(entries)
    # print(entries)
    print(length)
    for i in range(0, length):
        img_path = input_path + '/' + str(i) + image_format
        img = cv2.imread(img_path)
        img = cv2.resize(img, (w, h))
        save_path = image_preprocessing_output + '/' + str(i) + image_format
        cv2.imwrite(save_path, img)


    return length