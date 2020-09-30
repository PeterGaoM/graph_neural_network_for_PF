import os
import datetime
import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy.linalg import inv, pinv
import matplotlib.pyplot as plt
import scipy.io as scio
import scipy.stats as stats
from data_load import load_data_and_topology

from layers import GraphCon, ResNet50


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
        input_p = keras.Input(shape=(self.node_size, 1), name='input_p')
        input_q = keras.Input(shape=(self.node_size, 1), name='input_q')
        input_A_real = keras.Input(shape=(self.node_size, self.node_size), name='input_A_real')
        input_A_img = keras.Input(shape=(self.node_size, self.node_size), name='input_A_img')

        layer_p1, input_q1 = GraphCon(4, 128, activation='relu', name='layer_p1')([input_p, input_q, input_A_real, input_A_img])
        layer_p2, input_q2 = GraphCon(4, 128, activation='relu', name='layer_p2')([layer_p1, input_q1, input_A_real, input_A_img])
        layer_p3, input_q3 = GraphCon(4, 64, activation='relu', name='layer_p3')([layer_p2, input_q2, input_A_real, input_A_img])
        layer_p4, input_q4 = GraphCon(4, 16, activation='relu', name='layer_p4')([layer_p3, input_q3, input_A_real, input_A_img])
        layer_p5, input_q5 = GraphCon(4, 16, activation='relu', name='layer_p5')([layer_p4, input_q4, input_A_real, input_A_img])
        layer_p6, input_q6 = GraphCon(4, 4, activation='relu', name='layer_p6')([layer_p5, input_q5, input_A_real, input_A_img])
        layer_p7, input_q7 = GraphCon(4, 4, activation='relu', name='layer_p7')([layer_p6, input_q6, input_A_real, input_A_img])
        output_p, output_q = GraphCon(4, 1, activation='relu', name='output')([layer_p7, input_q7, input_A_real, input_A_img])

        self.model = keras.Model(inputs=[input_p, input_q, input_A_real, input_A_img],
                                 outputs=[output_p, output_p])
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
                traina = trainA[idx[from_idx:to_idx], :, :, :]
                train_p = trainx[:, :119, np.newaxis]
                train_q = trainx[:, 119:, np.newaxis]
                train_G = traina[:, :, :, 0]
                train_B = traina[:, :, :, 1]
                train_v = trainy[:, :119, np.newaxis]
                train_seta = trainy[:, 119:, np.newaxis]

                yield {'input_p': train_p, 'input_q': train_q, "input_A_real": train_G, "input_A_img": train_B}, \
                      {'output': train_v, "output":train_seta}
                      # {'output': np.concatenate([trainy[:, :self.node_size, np.newaxis], trainy[:, self.node_size:, np.newaxis]], axis=-1)}


    def load(self, model_dir):
        self.model.load_weights(model_dir)
        print("Load model:{}".format(model_dir))

    def predict(self, inputs, testA):
        res = self.model.predict([np.concatenate([inputs[:, :self.node_size, np.newaxis], inputs[:, self.node_size:, np.newaxis]], axis=-1), testA])
        return res


class Network_real_kernal():
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
        input_pq = keras.Input(shape=(self.node_size, 2), name='input_pq')
        input_A = keras.Input(shape=(self.node_size, self.node_size), name='input_A')
        output = ResNet50(input_pq, input_A)
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

    def train(self, train_images, train_labels, trainA, testData, testLabel, testA, epoch_num):
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
                       steps_per_epoch=2000,
                       epochs=epoch_num,
                       shuffle=True,
                       verbose=1,
                       callbacks=callbacks)

        self.model.save("./logs/model.h5")

    def data_generator(self, train_images, train_labels, trainA):
        while True:
            idx = np.random.permutation(train_images.shape[0])
            for k in range(int(np.ceil(train_images.shape[0] / self.batch_size))):
                from_idx = k * self.batch_size
                to_idx = (k + 1) * self.batch_size
                trainx = train_images[idx[from_idx:to_idx], :]
                trainy = train_labels[idx[from_idx:to_idx], :]
                traina = trainA[idx[from_idx:to_idx], :, :]
                yield {'input_pq': np.concatenate(
                    [trainx[:, :self.node_size, np.newaxis], trainx[:, self.node_size:, np.newaxis]], axis=-1),
                       'input_A': traina}, \
                      {'output': np.concatenate(
                          [trainy[:, :self.node_size, np.newaxis], trainy[:, self.node_size:, np.newaxis]], axis=-1)}

    def load(self, model_dir):
        self.model.load_weights(model_dir)
        print("Load model:{}".format(model_dir))

    def predict(self, inputs, testA):
        res = self.model.predict([np.concatenate([inputs[:, :self.node_size, np.newaxis], inputs[:, self.node_size:, np.newaxis]], axis=-1), testA])
        return res


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
    beta = 0
    fa = 0.01
    lr = 0.01
    batch_size = 20
    case_size = 188
    epoch_num = 500
    # 准备训练集
    trainData, trainLabel, trainA, testData, testLabel, testA = load_data_and_topology("../N-1/AC118/data_n-1_118.mat",
                                                                                       list(range(100)),
                                                                                       list(range(100, 180, 1)))
    # from data_load import load_data_AC14_initial
    # trainData, trainLabel, trainA, testData, testLabel, testA = load_data_AC14_initial(0.4)

    # 训练集处理
    # trainLabel[:, :case_size] = trainLabel[:, :case_size]**2
    trainData, mu1, sgma1 = Z_Score(trainData)
    trainLabel, mu, sgma = Z_Score(trainLabel)

    # testLabel[:, :case_size] = testLabel[:, :case_size] ** 2
    testData = (testData-mu1)/sgma1
    testData[np.isnan(testData)] = 0

    net = Network_real_kernal(beta, fa, lr, batch_size, case_size)
    # net.load('./logs/20200904T1552/model_0500.h5')
    # net.load('./logs/model.h5')
    net.train(trainData, trainLabel, trainA, testData, testLabel, testA, epoch_num)

    res = net.predict(testData, testA)
    # res = np.concatenate([res[:, :, 0], res[:, :, 1]], axis=1)
    # res = np.dot(res, pinv(A))
    # recover_res = res * sgma + mu
    # temp = (testLabel-mu)/sgma
    # temp[np.isnan(temp)] = 0
    res_recover = res * sgma + mu
    res_recover[:, :case_size] = res_recover[:, :case_size]**0.5
    deta = res_recover - testLabel
    # deta = (res - trainLabel)
    deta = np.abs(deta)
    deta1 = deta.copy()
    deta1 = deta1[:, case_size:]
    deta1[deta1 < 0.286] = 1
    deta1[deta1 != 1] = 0

    deta2 = deta.copy()
    deta2 = deta2[:, :case_size]
    deta2[deta2 < 0.0001] = 1
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





