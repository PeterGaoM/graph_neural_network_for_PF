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


class Linear_Output_Layer(keras.layers.Layer):
    def __init__(self, num_outputs, activation="linear", **kwargs):
        super(Linear_Output_Layer, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.activation_function = activation

    def build(self, input_shape):
        # Weights
        self.W = self.add_weight("W_",
                                  shape=(int(input_shape[0][-1]), self.num_outputs),
                                  initializer='random_normal',
                                  trainable=True)
        self.bias = self.add_weight("bias",
                                    shape=[self.num_outputs],
                                    initializer='random_normal', trainable=True)

    def call(self, inputs):
        input_ = inputs[0]
        y0 = keras.backend.dot(input_, self.W) + self.bias
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


def identity_block(input_tensor, input_A, iteration, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters

    bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = GraphCon(iteration, filters1, activation='linear', name=conv_name_base+'2a')([input_tensor, input_A])
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)

    x = GraphCon(iteration, filters2, activation='linear', name=conv_name_base+'2b')([x, input_A])
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)

    x = GraphCon(iteration, filters3, activation='linear', name=conv_name_base+'2c')([x, input_A])
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = keras.layers.add([x, input_tensor])
    x = keras.layers.Activation('relu')(x)
    return x


def conv_block(input_tensor, input_A, iteration, filters, stage, block):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = GraphCon(1, filters1, activation='linear', name=conv_name_base+'2a')([input_tensor, input_A])
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)

    x = GraphCon(iteration, filters2, activation='linear', name=conv_name_base + '2b')([x, input_A])
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)

    x = GraphCon(1, filters3, activation='linear', name=conv_name_base + '2c')([x, input_A])
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = GraphCon(1, filters3, activation='linear', name=conv_name_base + '1')([input_tensor, input_A])

    shortcut = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = keras.layers.add([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    return x


def ResNet50(input_tensor, input_A):

    bn_axis = 1

    x = GraphCon(7, 64, activation='linear', name='conv1')([input_tensor, input_A])
    x = keras.layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = keras.layers.Activation('relu')(x)

    x = conv_block(x, input_A, 4, [64, 64, 256], stage=2, block='a', )
    x = identity_block(x, input_A, 4, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, input_A, 4, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, input_A, 4, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, input_A, 4, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, input_A, 4, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, input_A, 4, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, input_A, 4, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, input_A, 4, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, input_A, 4, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, input_A, 4, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, input_A, 4, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, input_A, 4, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, input_A, 4, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, input_A, 4, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, input_A, 4, [512, 512, 2048], stage=5, block='c')

    output = Linear_Output_Layer(num_outputs=2, name='output')([x])

    return output


if __name__ == "__main__":
    node_size = 188
    input_pq = keras.Input(shape=(node_size, 2), name='input_pq')
    input_A = keras.Input(shape=(node_size, node_size), name='input_A')
    output = ResNet50(input_pq, input_A)
    #
    # # 浅层有功无功的特征提取
    # layer_pq1 = GraphCon(4, 128, activation='relu', name='layer_p1')([input_pq, input_A])
    # layer_pq2 = GraphCon(4, 128, activation='relu', name='layer_p2')([layer_pq1, input_A])
    # # layer_pq3 = GraphCon(4, 64, activation='relu', name='layer_p3')([layer_pq2, input_A])
    # layer_pq3 = GraphCon(4, 256, activation='relu', name='layer_p3')([layer_pq2, input_A])
    # # layer_pq3 = keras.layers.BatchNormalization()(layer_pq3)
    # layer_pq4 = GraphCon(4, 256, activation='relu', name='layer_p4')([layer_pq3, input_A])
    # layer_pq5 = GraphCon(4, 128, activation='relu', name='layer_p5')([layer_pq4, input_A])
    # layer_pq6 = GraphCon(4, 128, activation='relu', name='layer_p6')([layer_pq5, input_A])
    # layer_pq7 = GraphCon(4, 64, activation='relu', name='layer_p7')([layer_pq6, input_A])
    # layer_pq8 = GraphCon(4, 64, activation='relu', name='layer_p8')([layer_pq7, input_A])
    # Linear_Output_Layer(2, activation='linear', name='layer_p8')([layer_pq8])
    input_pq = keras.Input(shape=(299,299, 1024), name='input_pq')


