import cv2
import numpy as np
import matplotlib.pyplot as plt


def face_cut(path, name):
    global cut_image
    PADDING = 0
    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    filename = 'D:/untitled/Face-Recognition-master/test/' + name + '.jpg'

    print(filename)
    img = cv2.imread(filename)
    frame = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_coord = face.detectMultiScale(gray, 1.2, 7, minSize=(50, 50))

    faces = cutfaces(img, faces_coord)

    if len(faces) != 0:

        # cv2.imwrite('img_test.jpg',faces[0])

        for (x, y, w, h) in faces_coord:
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            img = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            height, width, channels = frame.shape
            cut_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
            cv2.imwrite(path + name + ",.jpg", cut_image)
            #print(path)


            # left_cheek
            x11 = x
            y11 = int(y + 0.3*h)
            x21 = int(x + w/2)
            y21 = int(y + 0.85*h)

            img = cv2.rectangle(frame, (x11, y11), (x21, y21), (255, 255, 255), 2)
            height, width, channels = frame.shape
            cut_image = frame[max(0, y11):min(height, y21), max(0, x11):min(width, x21)]
            print(path)
            cv2.imwrite(path + name + ",left_cheek.jpg", cut_image)
            #cv2.imshow('img', cut_image)
            #k = cv2.waitKey(730) & 0xff


            # right_cheek
            x12 = int(x + w/2)
            y12 = int(y + 0.3*h)
            x22 = int(x + w)
            y22 = int(y + 0.85*h)

            img = cv2.rectangle(frame, (x12, y12), (x22, y22), (255, 255, 255), 2)
            height, width, channels = frame.shape
            cut_image = frame[max(0, y12):min(height, y22), max(0, x12):min(width, x22)]
            #print(path)
            cv2.imwrite(path + name + ",right_cheek.jpg", cut_image)

            # forehead
            x13 = x
            y13 = y
            x23 = int(x + w)
            y23 = int(y + 0.3*h)

            img = cv2.rectangle(frame, (x13, y13), (x23, y23), (255, 255, 255), 2)
            height, width, channels = frame.shape
            cut_image = frame[max(0, y13):min(height, y23), max(0, x13):min(width, x23)]
            #print(path)
            cv2.imwrite(path + name + ",forehead.jpg", cut_image)

            break

    # cv2.imshow('img', img)
    # k = cv2.waitKey(30) & 0xff

    #    webcam.release()
    plt.imshow(img)
    cv2.destroyAllWindows()
    return cut_image


def cutfaces(image, faces_coord):
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


if __name__ == '__main__':
    PADDING = 25
    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread('temp1.JPG', 1)
    frame = img
    #frame = np.asarray(frame, dtype=np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_coord = face.detectMultiScale(gray, 1.2, 7, minSize=(50, 50))
    faces = cutfaces(img, faces_coord)

    if (len(faces) != 0):

        # cv2.imwrite('img.jpg',faces[0])

        for (x, y, w, h) in faces_coord:
            x1 = x - PADDING
            y1 = y - PADDING
            x2 = x + w + PADDING
            y2 = y + h + PADDING

            img = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            height, width, channels = frame.shape
            part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
        cv2.imwrite('temp.jpg', part_image)
