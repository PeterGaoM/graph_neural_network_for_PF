import os
import datetime
import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy.linalg import inv, pinv
import matplotlib.pyplot as plt
import scipy.io as scio
import scipy.stats as stats


class NewNetwork():
    def __init__(self, beta, fa, lr, batch_size, node_size):
        self.beta = beta
        self.lr = lr
        self.batch_size = batch_size
        self.fa = fa
        self.model = None
        self.node_size = node_size
        self.build_model()

    def new_loss(self, y_true, y_pred):
        var = keras.losses.mse(y_true, y_pred)
        # var = tf.math.reduce_variance(y_pred)
        loss = 1/(var + 1)
        return loss

    def new_loss2(self, y_true, y_pred):
        loss1 = keras.losses.mse(y_true, y_pred)
        loss2 = tf.math.reduce_max(y_true-y_pred)
        # loss3 = (y_true-y_pred)*tf.math.log(y_true-y_pred)
        return loss2 + loss1

    def build_model(self):
        # 定义网络的输入
        input_pq = keras.Input(shape=(self.node_size, 2), name='input_pq')
        input_A = keras.Input(shape=(self.node_size, self.node_size), name='input_A')

        # 浅层有功无功的特征提取
        layer_pq1 = GraphCon(4, 128, activation='relu', name='layer_p1')([input_pq, input_A])
        layer_pq2 = GraphCon(4, 128, activation='relu', name='layer_p2')([layer_pq1, input_A])
        layer_pq3 = GraphCon(4, 64, activation='relu', name='layer_p3')([layer_pq2, input_A])
        # layer_pq3 = GraphCon(4, 256, activation='relu', name='layer_p3')([layer_pq2, input_A])
        # # layer_pq3 = keras.layers.BatchNormalization()(layer_pq3)
        # layer_pq4 = GraphCon(4, 256, activation='relu', name='layer_p4')([layer_pq3, input_A])
        # layer_pq5 = GraphCon(4, 128, activation='relu', name='layer_p5')([layer_pq4, input_A])
        # layer_pq6 = GraphCon(4, 128, activation='relu', name='layer_p6')([layer_pq5, input_A])
        # layer_pq7 = GraphCon(4, 128, activation='relu', name='layer_p7')([layer_pq6, input_A])
        # layer_pq8 = GraphCon(4, 128, activation='relu', name='layer_p8')([layer_pq7, input_A])
        # output = GraphCon(1, 2, activation='linear', name='output')([layer_pq2, input_A])

        output1 = keras.layers.Flatten()(layer_pq3)
        # # x3 = keras.layers.BatchNormalization()(x3)
        output2 = keras.layers.Dense(200, activation='relu')(output1)
        output = keras.layers.Dense(self.node_size * 2, activation='linear', name='output')(output2)
        # self.model = keras.Model(inputs=[input_pq, input_A], outputs=[output, layer_pq5, layer_pq4, layer_pq3, layer_pq2, layer_pq1])
        self.model = keras.Model(inputs=[input_pq, input_A],
                                 outputs=[output])
        optimizer = keras.optimizers.Adam(lr=self.lr, decay=1e-6)
        # optimizer = keras.optimizers.SGD(lr=self.lr, decay=1e-6, momentum=0.8)
        self.model.compile(optimizer=optimizer,
                           loss='mse',
                           metrics=["accuracy"])
        # sgd = keras.optimizers.SGD(lr=self.lr)
        # self.model.compile(loss=self.new_loss, optimizer=sgd, metrics=["accuracy"])
        # # self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
        self.model.summary()
        keras.utils.plot_model(self.model, 'model_graph.png', show_shapes=True)

    def train(self, train_images,  train_labels, trainA, epoch_num):
        now = datetime.datetime.now()
        self.log_dir = os.path.join('./logs', "{:%Y%m%dT%H%M}".format(now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "model_*epoch*.h5")
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")
        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]
        data_gen = self.data_generator(train_images, train_labels, trainA)
        self.model.fit(data_gen,
                       steps_per_epoch=3000,
                       epochs=epoch_num,
                       shuffle=True,
                       verbose=1,
                       callbacks=callbacks)

        self.model.save("./logs/model.h5")

    def data_generator(self,  train_images,  train_labels, trainA):
        while True:
            idx = np.random.permutation(train_images.shape[0])
            for k in range(int(np.ceil(train_images.shape[0] / self.batch_size))):
                from_idx = k * self.batch_size
                to_idx = (k + 1) * self.batch_size
                trainx = train_images[idx[from_idx:to_idx], :]
                trainy = train_labels[idx[from_idx:to_idx], :]
                traina = trainA[idx[from_idx:to_idx], :, :]
                yield {'input_pq': np.concatenate([trainx[:, :self.node_size, np.newaxis], trainx[:, self.node_size:, np.newaxis]], axis=-1),
                       'input_A': traina}, \
                      {'output': trainy}
                      # {'output': np.concatenate([trainy[:, :self.node_size, np.newaxis], trainy[:, self.node_size:, np.newaxis]], axis=-1)}


    def load(self, model_dir):
        self.model.load_weights(model_dir)
        print("Load model:{}".format(model_dir))

    def predict(self, inputs, testA):
        res = self.model.predict([np.concatenate([inputs[:, :self.node_size, np.newaxis], inputs[:, self.node_size:, np.newaxis]], axis=-1), testA])
        return res


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


def load_data_AC14_initial(ratio):
    # 获取各种短线的index
    dataFile = '../N-1/AC14/data_none_cut.mat'
    data = scio.loadmat(dataFile)
    dataX = data['conved_Y_load']
    dataY = data['gotten_VA']
    dataY2 = data['gotten_VM']

    # 读取线路连接关系，获得节点连接矩阵
    branch = scio.loadmat('../N-1/AC14/structureY.mat')
    structure = branch['Y']
    # for x in range(structure.shape[0]):
    #     structure[x, x] = 0
    D = np.diag(np.sum(structure, axis=0)**(-0.5))
    structure = np.dot(np.dot(D, structure), D)
    # branch = branch['branch']

    trainData = dataX[:int(dataX.shape[0]*ratio), :]
    trainLabel = np.concatenate([dataY2[:int(dataX.shape[0]*ratio), :], dataY[:int(dataX.shape[0]*ratio), :]], axis=1)
    trainA = []
    for num in range(trainData.shape[0]):
        trainA.append(structure)

    testData = dataX[int(dataX.shape[0]*ratio):, :]
    testLabel = np.concatenate([dataY2[int(dataX.shape[0]*ratio):, :], dataY[int(dataX.shape[0]*ratio):, :]], axis=1)
    testA = []
    for num in range(testData.shape[0]):
        testA.append(structure)

    return np.array(trainData), np.array(trainLabel), np.array(trainA), \
           np.array(testData), np.array(testLabel), np.array(testA)


def load_data_AC14_topo1(ratio):
    # 获取各种短线的index
    dataFile = '../N-1/AC14/data_cut1.mat'
    data = scio.loadmat(dataFile)
    dataX = data['conved_Y_load']
    dataY = data['gotten_VA']
    dataY2 = data['gotten_VM']

    # 读取线路连接关系，获得节点连接矩阵
    branch = scio.loadmat('../N-1/AC14/structure_cut1_Y.mat')
    structure = branch['Y']
    D = np.diag(np.sum(structure, axis=0)**(-0.5))
    structure = np.dot(np.dot(D, structure), D)
    # branch = branch['branch']

    trainData = dataX[:int(dataX.shape[0]*ratio), :]
    trainLabel = np.concatenate([dataY2[:int(dataX.shape[0]*ratio), :], dataY[:int(dataX.shape[0]*ratio), :]], axis=1)
    trainA = []
    for num in range(trainData.shape[0]):
        trainA.append(structure)

    testData = dataX[int(dataX.shape[0]*ratio):, :]
    testLabel = np.concatenate([dataY2[int(dataX.shape[0]*ratio):, :], dataY[int(dataX.shape[0]*ratio):, :]], axis=1)
    testA = []
    for num in range(testData.shape[0]):
        testA.append(structure)

    return np.array(trainData), np.array(trainLabel), np.array(trainA), \
           np.array(testData), np.array(testLabel), np.array(testA)


def load_data_AC14_topo2(ratio):
    # 获取各种短线的index
    dataFile = '../N-1/AC14/data_cut2.mat'
    data = scio.loadmat(dataFile)
    dataX = data['conved_Y_load']
    dataY = data['gotten_VA']
    dataY2 = data['gotten_VM']

    # 读取线路连接关系，获得节点连接矩阵
    branch = scio.loadmat('../N-1/AC14/structure_cut2_Y.mat')
    structure = branch['Y']
    D = np.diag(np.sum(structure, axis=0)**(-0.5))
    structure = np.dot(np.dot(D, structure), D)
    # branch = branch['branch']

    trainData = dataX[:int(dataX.shape[0]*ratio), :]
    trainLabel = np.concatenate([dataY2[:int(dataX.shape[0]*ratio), :], dataY[:int(dataX.shape[0]*ratio), :]], axis=1)
    trainA = []
    for num in range(trainData.shape[0]):
        trainA.append(structure)

    testData = dataX[int(dataX.shape[0]*ratio):, :]
    testLabel = np.concatenate([dataY2[int(dataX.shape[0]*ratio):, :], dataY[int(dataX.shape[0]*ratio):, :]], axis=1)
    testA = []
    for num in range(testData.shape[0]):
        testA.append(structure)

    return np.array(trainData), np.array(trainLabel), np.array(trainA), \
           np.array(testData), np.array(testLabel), np.array(testA)


def Z_Score(trainData):
    trainData = np.array(trainData)
    mean_train = np.mean(trainData, axis=0)
    std_train = np.std(trainData, axis=0)
    std_train[std_train<0.0001] = 0
    trainData = (trainData - mean_train) / std_train
    trainData[np.isnan(trainData)] = 0
    trainData[np.isinf(trainData)] = 0
    return trainData, mean_train, std_train


if __name__ == "__main__":
    # 准备训练集
    trainData0, trainLabel0, trainA0, testData0, testLabel0, testA0 = load_data_AC14_initial(0.9)
    trainData1, trainLabel1, trainA1, testData1, testLabel1, testA1 = load_data_AC14_topo1(0.9)
    trainData2, trainLabel2, trainA2, testData2, testLabel2, testA2 = load_data_AC14_topo2(0.9)

    trainData = np.concatenate([trainData0, trainData2[:5, :]], axis=0)
    trainLabel = np.concatenate([trainLabel0, trainLabel2[:5, :]], axis=0)
    trainA = np.concatenate([trainA0, trainA2[:5, :]], axis=0)

    # 训练集处理
    trainLabel[:, :14] = trainLabel[:, :14]**2
    trainData, mu1, sgma1 = Z_Score(trainData)
    trainLabel, mu, sgma = Z_Score(trainLabel)
    trainA = trainA

    testLabel = testLabel0
    # testLabel[:, :14] = testLabel[:, :14] ** 2
    testData = (testData0-mu1)/sgma1
    testData[np.isnan(testData)] = 0
    testA = testA0

    beta = 0
    fa = 0.01
    lr = 0.0001
    batch_size = 200
    net = NewNetwork(beta, fa, lr, batch_size, 14)
    # net.load('./logs/20200904T1552/model_0500.h5')
    net.load('./logs/model.h5')
    net.train(trainData, trainLabel, trainA, epoch_num=5)

    res = net.predict(testData, testA)
    # res = np.concatenate([res[:, :, 0], res[:, :, 1]], axis=1)
    # res = np.dot(res, pinv(A))
    # recover_res = res * sgma + mu
    # temp = (testLabel-mu)/sgma
    # temp[np.isnan(temp)] = 0
    res_recover = res * sgma + mu
    res_recover[:, :14] = res_recover[:, :14]**0.5
    deta = res_recover - testLabel
    # deta = (res - trainLabel)
    deta = np.abs(deta)
    deta1 = deta.copy()
    deta1 = deta1[:, 14:]
    deta1[deta1 < 0.286] = 1
    deta1[deta1 != 1] = 0

    deta2 = deta.copy()
    deta2 = deta2[:, :14]
    deta2[deta2 < 0.001] = 1
    deta2[deta2 != 1] = 0

    print(np.mean(deta1), np.mean(deta2))
    # print(np.mean(det:a[1, :118]**2), np.mean(deta[1, 118:]**2))
    # print(np.var(deta[1, :118]), np.var(deta[1, 118:]))

    # plt.subplot(2, 1, 1)
    # plt.plot(deta[1, :118])
    # plt.subplot(2, 1, 2)
    # plt.plot(deta[1, 118:])
    # plt.show()
    #
    # deta = deta[:, 118:]
    # deta[deta < 0.2865] = 1
    # deta[deta != 1] = 0
    # print(np.mean(deta))

    # # 不进行迁移，或者迁移之后不训练第二种拓扑，直接预测第二种拓扑
    # net.train(datax1, datay1, epoch_num=200)
    # # net.load('./logs/model_[199].h5')
    # _, res = net.predict(datax2)
    # recover_res = res * sgma + mu
    # deta = res * sgma - datay2 * sgma
    # result = np.abs(np.mean(deta, axis=1))
    # result[result < 0.2865] = 1
    # result[result != 1] = 0
    # print(np.mean(result))





