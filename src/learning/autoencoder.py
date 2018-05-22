import math

import numpy as np
import tensorflow as tf

import data


class Autoencoder(object):

    def __init__(self, n_layers, n_features, x):
        self.n_layers = n_layers
        self.n_features = n_features
        self.x = x

        self._build_model()

    def _build_model(self):
        layer = self.x
        layer = self._build_encoder(layer)
        predictions = self._build_decoder(layer)

        with tf.name_scope("train"):
            self.loss = tf.losses.mean_squared_error(
                self.x,
                predictions)
            tf.summary.scalar("loss", self.loss)

            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        self.summary = tf.summary.merge_all()

    def _build_encoder(self, layer):
        # Encode
        with tf.variable_scope('encoding'):
            self.encoder_layers = []
            for layer_units in self.n_layers:
                layer = tf.layers.dropout(
                    inputs=layer,
                    rate=0.7,
                    name="dropout_{}".format(layer_units))

                layer = tf.layers.dense(
                    inputs=layer,
                    units=layer_units,
                    activation=tf.nn.relu,
                    name="dense_{}".format(layer_units))

                self.encoder_layers.append(layer)

            # Put hidden weights in [-1, 1] so normal random samples can be
            # inserted
            layer = tf.tanh(layer)

            return layer

    def _build_decoder(self, layer):
        # Decode
        with tf.variable_scope('decoding'):
            self.decoder_layers = []
            for layer_units in list(reversed(self.n_layers))[1:] + [self.n_features]:
                layer = tf.layers.dropout(
                    inputs=layer,
                    rate=0.3,
                    name="dropout_{}".format(layer_units))

                layer = tf.layers.dense(
                    inputs=layer,
                    units=layer_units,
                    name="dense_{}".format(layer_units))

                self.decoder_layers.append(layer)

            predictions = self.decoder_layers[-1]
            return predictions

    def partial_fit(self, sess):
        loss, summary, _ = sess.run(
            (self.loss, self.summary, self.optimizer))
        return loss, summary

    def decode(self, sessl, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.n_layers[-1])
        return sess.run((self._build_decoder(hidden),
                         self.summary))


if __name__ == '__main__':
    poses = data.get_poses()
    np.random.shuffle(poses)
    print(poses.shape)
    n_poses = poses.shape[0]
    n_poses_train = math.ceil(n_poses * 0.8)

    SUMMARIES_DIR = 'autoencoder_logs'

    poses_train = poses[:n_poses_train]
    poses_test = poses[n_poses_train:]

    train_data = (
        tf.data.Dataset
        .from_tensor_slices((poses_train,))
        .repeat()
        .shuffle(1024)
        .batch(128))

    test_data = (
        tf.data.Dataset
        .from_tensor_slices(poses_train)
        .batch(1))
    test_iter = test_data.make_one_shot_iterator()

    with tf.Session() as sess:
        with tf.name_scope("input"):
            train_iter = train_data.make_one_shot_iterator()
            x = tf.reshape(train_iter.get_next(), [-1, 30])

        autoencoder = Autoencoder(
            n_features=30,
            n_layers=[10, 4],
            x=x)

        train_writer = tf.summary.FileWriter(
            SUMMARIES_DIR + '/train',
            sess.graph)

        sess.run(tf.global_variables_initializer())

        for i in range(1000):
            try:
                loss, summary = autoencoder.partial_fit(sess)
                train_writer.add_summary(summary, i)
            except tf.errors.OutOfRangeError:
                break
