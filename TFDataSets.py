from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


# ------------------------------- MAIN ------------------------------- #
if __name__ == '__main__':
    tf.enable_v2_behavior()
    # tfds.disable_progress_bar()
    # print(tfds.list_builders())

    mnist_train, info = tfds.load(name="mnist",
                                  split="train",
                                  data_dir="~/Documents/Python/ML-recipes/Data/tf_datasets",
                                  with_info=True,
                                  download=False)
    assert isinstance(mnist_train, tf.data.Dataset)
    print(mnist_train)

    # same as above
    # mnist_builder = tfds.builder("mnist")
    # mnist_builder.download_and_prepare()
    # mnist_train = mnist_builder.as_dataset(split="train")
    # print(mnist_train)

    for mnist_example in mnist_train.take(1):  # Only take a single example
        image, label = mnist_example["image"], mnist_example["label"]

        plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap("Blues"))
        plt.title("Digit: %d" % label.numpy())

    print(info)

    mnist_test, info = tfds.load(name="mnist",
                                 split="test",
                                 data_dir="~/Documents/Python/ML-recipes/Data/tf_datasets",
                                 with_info=True,
                                 download=False)

    fig = tfds.show_examples(info, mnist_test)
    plt.show()
# ------------------------------- MAIN ------------------------------- #
