from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import tensorflow as tf


class CustomTraining(object):
    def __init__(self):
        self.TRUE_W = tf.Variable(3.0)
        self.TRUE_b = tf.Variable(2.0)
        self.NUM_EXAMPLES = 1000

        self.W = tf.Variable(tf.random.normal(shape=[1]))
        self.b = tf.Variable(tf.random.normal(shape=[1]))

        noise = tf.random.normal(shape=[self.NUM_EXAMPLES])

        self.inputs = tf.random.normal(shape=[self.NUM_EXAMPLES])
        self.outputs = self.inputs * self.TRUE_W + self.TRUE_b + noise

    def __call__(self, x):
        return self.W * x + self.b

    def show(self):
        plt.title("True+noise IO (BLUE) Predicted IO (RED)")
        plt.scatter(self.inputs, self.outputs, c='b')
        plt.scatter(self.inputs, self.__call__(self.inputs), c='r')
        plt.show()
        print('Current loss: %1.6f' % self.loss(self.__call__(self.inputs), self.outputs).numpy())

    def training_step(self, learning_rate):
        with tf.GradientTape() as t:
            current_loss = self.loss(self.__call__(self.inputs), self.outputs)
        dw, db = t.gradient(current_loss, [self.W, self.b])
        self.W.assign_sub(learning_rate * dw)
        self.b.assign_sub(learning_rate * db)

    def train(self, learning_rate=0.05, epochs=range(60)):

        # Collect the history of W-values and b-values to plot later
        ws, bs = [], []
        for epoch in epochs:
            ws.append(self.W.numpy())
            bs.append(self.b.numpy())
            current_loss = self.loss(self.__call__(self.inputs), self.outputs)

            self.training_step(learning_rate)
            print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' % (epoch, ws[-1], bs[-1], current_loss))

        # Let's plot it all
        plt.plot(epochs, ws, 'r', epochs, bs, 'b')
        plt.plot([self.TRUE_W.numpy()] * len(epochs), 'r--', [self.TRUE_b.numpy()] * len(epochs), 'b--')
        plt.legend(['W', 'b', 'True W', 'True b'])
        plt.show()

    @staticmethod
    def loss(predicted_y, target_y):
        return tf.reduce_mean(tf.square(predicted_y - target_y))

    @staticmethod
    def demo():
        v = tf.Variable(1.0)
        # Use Python's `assert` as a debugging statement to test the condition
        assert v.numpy() == 1.0

        # Reassign the value `v`
        v.assign(3.0)
        assert v.numpy() == 3.0

        # Use `v` in a TensorFlow `tf.square()` operation and reassign
        v.assign(tf.square(v))
        assert v.numpy() == 9.0

        x = tf.zeros([10, 10])
        x += 2
        print(x)


# ------------------------------- MAIN ------------------------------- #
if __name__ == '__main__':
    model = CustomTraining()
    # model.demo()
    # model.show()
    model.train()
