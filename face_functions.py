import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

from fr_utils import img_to_encoding


def cutfaces(image, faces_coord):
    faces = []

    for (x, y, w, h) in faces_coord:
        w_rm = int(0.2 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])

    return faces


def cutfaces1(image, faces_coord):
    faces = []

    for (x, y, w, h) in faces_coord:
        w_rm = int(0.2 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])

    return faces


def normalize_histogram(images):
    face_norm = []
    for image in images:
        face_norm.append(cv2.equalizeHist(image))
    return face_norm


def normalize_image(image):
    alpha = 1.3
    beta = 25

    new_image = np.zeros(image.shape)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):

            for c in range(image.shape[2]):
                new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)

    return new_image


def resize_image(image, size=(96, 96)):
    if image.shape < size:
        image_resize = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    else:
        image_resize = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

    return image_resize


def prepare_database(model):
    database = {}
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_to_encoding(file, model)

    return database


def prepare_database1(model):
    database = {}
    for file in glob.glob("diseases/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_to_encoding(file, model)

    return database


def add_to_database(name, folder_directory,pname):
    # name += '.jpg'
    name = name[0:len(name) - 4]
    path = os.path.join('images', name)
    image = webcam(path, name, folder_directory,pname);


def add_to_database2(name, folder_directory,pname):
    # name += '.jpg'
    name = name[0:len(name) - 4]
    path = os.path.join('test', name)
    image = webcam(path, name, folder_directory,pname);


def webcam(path, name, folder_directory,pname):
    global cut_image
    PADDING = 25
    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # webcam = cv2.VideoCapture(0)
    count = 0
    while True:
        count += 1
        if (count == 2):
            break

        filename = folder_directory + name + '.jpg'

        print(filename)
        img = cv2.imread(filename)
        frame = img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_coord = face.detectMultiScale(gray, 1.2, 7, minSize=(50, 50))

        faces = cutfaces(img, faces_coord)

        if (len(faces) != 0):

            # cv2.imwrite('img_test.jpg',faces[0])

            for (x, y, w, h) in faces_coord:
                x1 = x - PADDING
                y1 = y - PADDING
                x2 = x + w + PADDING
                y2 = y + h + PADDING
                img = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                height, width, channels = frame.shape
                cut_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                print(path)
                cv2.imwrite(path + ',' + pname + ",.jpg", cut_image)
            break

        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    #    webcam.release()
    if not cut_image.any() and count == 2:
        return 0

    plt.imshow(img)
    cv2.destroyAllWindows()
    return cut_image


def speak(text, rate=2):
    # speak = win32com.client.Dispatch('Sapi.SpVoice')
    # speak.Voice =  speak.GetVoices('Microsoft Zira')
    speak.Volume = 100
    speak.Rate = rate
    speak.Speak(text)


def recognise_face(imagepath: object, database: object, model: object) -> object:
    encoding = img_to_encoding(imagepath, model)
    identity = None
    min_dist = 100

    for (name, db_enc) in database.items():

        dist = np.linalg.norm(db_enc - encoding)

        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.5:

        return str(0)

    else:
        return str(identity)


def recognise_face1(imagepath: object, database: object, model: object) -> object:
    encoding = img_to_encoding(imagepath, model)
    identity = None
    min_dist = 100

    for (name, db_enc) in database.items():

        dist = np.linalg.norm(db_enc - encoding)

        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.4:

        return str(0)

    else:
        return str(identity)


a1 = np.empty(70, dtype=int)


def recognise_face2(imagepath: object, database: object, model: object) -> object:
    encoding = img_to_encoding(imagepath, model)

    i = 0

    for (name, db_enc) in database.items():
        dist1 = np.linalg.norm(db_enc - encoding)
        a1[i] = dist1
        i = i + 1
        print(dist1)

    return a1
