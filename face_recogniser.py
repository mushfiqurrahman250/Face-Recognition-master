import csv
from fr_utils import *
from inception_network import *
from face_functions import *

K.set_image_data_format('channels_first')


def triplet_loss_function(y_true, y_pred, alpha=0.3):
    anchor = y_pred[0]
    positive = y_pred[1]
    negative = y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss


if __name__ == '__main__':

    # speak('compiling Model.....', 1)
    model = model(input_shape=(3, 96, 96))
    model.compile(optimizer='adam', loss=triplet_loss_function, metrics=['accuracy'])
    # speak('model compile successful', 1)
    # speak('loading weights into model, this might take sometime sir!', 1)
    load_weights_from_FaceNet(model)
    # while True:
    # speak('model ready to roll sir!')
    # decision = input("Initiate face_recognition sequence press Y/N: ")
    # decision = sys.argv[1]
    # if decision == ('y' or 'Y'):
    # images = []
    database = prepare_database(model)
    i = 1
    for filename in os.listdir("test"):
        path = os.path.join("test", filename)
        x = filename.split(",", 3)

        print(x)

        face = recognise_face(path, database, model)
        # face = recognise_face1(path, database, model)
        y = face.split(",", 3)

        print(y)
        print("\n")

        with open('face.csv', 'a', newline='') as fs:
            a = csv.writer(fs, delimiter=",")
            data = []
            if len(x) >= 2 and len(y) >= 2 and (x[1] == y[1] or x[2] == y[2]):
                filename = x[1]
                p = y[1]

            elif y[0] == '0':
                p = 0
                filename = 0

            else:
                p = y[1]
                filename = x[1]

            data = [x, y, p, filename]
            a.writerow(data)



