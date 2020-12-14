import math
from os import walk

from face_functions import speak, add_to_database, add_to_database2

import cv2
import os


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
        return images


directory = "C:/Users/ASUS/Documents/aligned_images_DB/"
entries = os.listdir(directory)
person =-1
for mydirectory in os.walk(directory):

    for i in range(0, len(mydirectory[1])):
        new_directory = directory + mydirectory[1][i]
        for mypath in os.walk(new_directory):
            print(mypath)
            for j in range(0, len(mypath[1])):
                folder_directory = new_directory + '/' + mypath[1][j] + '/'
                print(folder_directory)
                for (_, _, filenames) in walk(folder_directory):
                    for k in range(0, math.ceil(len(filenames) / 100 * 70)):
                        add_to_database(filenames[k], folder_directory, entries[i])

                    for m in range(k + 1, len(filenames)):
                        if filenames[m] != "":
                            add_to_database2(filenames[m], folder_directory,entries[i])
                    break
