import os
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class TFSoftmaxClassifier:

    def __init__(self):
        tf.enable_v2_behavior()
        self.data_full_path = os.path.join(os.path.normpath(os.path.dirname(__file__)), "Data", "tf_datasets")
        print(self.data_full_path)

        try:
            self.mnist_train = tfds.load(name="mnist",
                                         split="train",
                                         data_dir=self.data_full_path,
                                         with_info=False,
                                         download=False)

            self.mnist_test, self.mnist_info = tfds.load(name="mnist",
                                                         split="test",
                                                         data_dir=self.data_full_path,
                                                         with_info=True,
                                                         download=False)
        except AssertionError as e:
            print(e)
            os.makedirs(self.data_full_path, 0o777)
            self.mnist_train = tfds.load(name="mnist",
                                         split="train",
                                         data_dir=self.data_full_path,
                                         with_info=False,
                                         download=True)

            self.mnist_test, self.mnist_info = tfds.load(name="mnist",
                                                         split="test",
                                                         data_dir=self.data_full_path,
                                                         with_info=True,
                                                         download=True)

    def fit(self):
        pass

    def predict(self):
        pass

    def info(self):
        print(self.mnist_info)

        for mnist_example in self.mnist_train.take(1):  # Only take a single example
            image, label = mnist_example["image"], mnist_example["label"]

            plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap("Blues"))
            plt.title("Digit: %d" % label.numpy())

            # x = self.mnist_info.features["image"].shape[0]
            # y = self.mnist_info.features["image"].shape[1][1]

            print(mnist_example["image"][0][4])
            # XX = tf.reshape(X, [-1, 784])

            # for

            # print("number of data points: ", self.mnist_train.images.shape[0],
            #       "number of pixels in each image:", self.mnist_train.images.shape[1])
            tfds.show_examples(self.mnist_info, self.mnist_test)
            plt.show()


# ------------------------------- MAIN ------------------------------- #
if __name__ == '__main__':
    tf_clf = TFSoftmaxClassifier()
    tf_clf.info()
