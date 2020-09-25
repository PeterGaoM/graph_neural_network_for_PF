import tensorflow as tf
from tensorflow import keras


class GraphCon(keras.layers.Layer):
    def __init__(self, iteration, num_outputs, activation="sigmoid", **kwargs):
        super(GraphCon, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.activation_function = activation
        self.iteration = iteration

    def build(self, input_shape):
        # Weights
        # temp = input_shape[0]
        self.W = {}
        for n in range(self.iteration):
            self.W["W_"+str(n)] = self.add_weight("W_"+str(n),
                                  shape=(int(input_shape[0][-1]), self.num_outputs),
                                  initializer='random_normal',
                                  trainable=True)
        # self.bias = self.add_weight("bias",
        #                             shape=[self.num_outputs],
        #                             initializer='random_normal', trainable=True)

    def call(self, inputs):
        input_ = inputs[0]
        A = inputs[1]
        y0 = keras.backend.dot(input_, self.W["W_0"])
        for n in range(1, self.iteration):
            input_ = keras.backend.batch_dot(A, input_)
            y_temp = keras.backend.dot(input_, self.W["W_"+str(n)])
            y0 = tf.keras.layers.Add()([y0, y_temp])
        if self.activation_function == 'relu':
            return keras.backend.relu(y0)
        elif self.activation_function == "linear":
            return y0
        else:
            return keras.backend.sigmoid(y0)


class GraphCon_complex_kernal(keras.layers.Layer):
    """
    转移矩阵连接为复数矩阵
    """
    def __init__(self, iteration, num_outputs, activation="sigmoid", **kwargs):
        super(GraphCon_complex_kernal, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.activation_function = activation
        self.iteration = iteration

    def build(self, input_shape):
        # Weights
        # temp = input_shape[0]
        self.W = {}
        for n in range(self.iteration):
            self.W["W_real_"+str(n)] = self.add_weight("W_real_"+str(n),
                                  shape=(int(input_shape[0][-1]), self.num_outputs),
                                  initializer='random_normal',
                                  trainable=True)
            self.W["W_img_" + str(n)] = self.add_weight("W_img_" + str(n),
                                                    shape=(int(input_shape[0][-1]), self.num_outputs),
                                                    initializer='random_normal',
                                                    trainable=True)
        # self.bias_real = self.add_weight("bias_real",
        #                                  shape=[self.num_outputs],
        #                                  initializer='random_normal', trainable=True)
        # self.bias_img = self.add_weight("bias_img",
        #                                 shape=[self.num_outputs],
        #                                 initializer='random_normal', trainable=True)

    def call(self, inputs):
        input_real = inputs[0]
        input_img = inputs[1]
        A_real = inputs[2]
        A_img = inputs[3]
        y0_real = tf.keras.layers.Add()([keras.backend.dot(input_real, self.W["W_real_0"]),
                                        -keras.backend.dot(input_img, self.W["W_img_0"])])
        y0_img = tf.keras.layers.Add()([keras.backend.dot(input_img, self.W["W_real_0"]),
                                        keras.backend.dot(input_real, self.W["W_img_0"])])

        for n in range(1, self.iteration):
            input_real = tf.keras.layers.Add()([keras.backend.batch_dot(A_real, input_real),
                                                -keras.backend.batch_dot(A_img, input_img)])
            input_img = tf.keras.layers.Add()([keras.backend.batch_dot(A_real, input_img),
                                               keras.backend.batch_dot(A_img, input_real)])

            y0_temp_real = tf.keras.layers.Add()([keras.backend.dot(input_real, self.W["W_real_"+str(n)]),
                                                  -keras.backend.dot(input_img, self.W["W_img_"+str(n)])])
            y0_temp_img = tf.keras.layers.Add()([keras.backend.dot(input_img, self.W["W_real_"+str(n)]),
                                                 keras.backend.dot(input_real, self.W["W_img_"+str(n)])])

            y0_real = tf.keras.layers.Add()([y0_real, y0_temp_real])
            y0_img = tf.keras.layers.Add()([y0_img, y0_temp_img])
        # y0_real = y0_real + self.bias_real
        # y0_img = y0_img + self.bias_img
        if self.activation_function == 'relu':
            return keras.backend.relu(y0_real), keras.backend.relu(y0_img)
        elif self.activation_function == "linear":
            return y0_real, y0_img
        else:
            return keras.backend.sigmoid(y0_real), keras.backend.sigmoid(y0_img)

