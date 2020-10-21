from keras import backend as K
K.set_image_data_format('channels_first')
from inception_network import *
from face_functions import *


l = 0
for filename in os.listdir("test"):
    l = l + 1
print(l)
a = np.empty(l, dtype=int)
b = np.empty(l, dtype=int)
i = 0


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
    model.compile(optimizer='adam', loss=triplet_loss_function, metrics=['accuracy'])

    database = prepare_database(model)

    for filename in os.listdir("test"):
        path = os.path.join("test", filename)
        print(filename)
        face = recognise_face2(path, database, model)
        print("\n")




