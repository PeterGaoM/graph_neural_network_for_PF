import numpy as np
from numpy.linalg import inv, pinv
import matplotlib.pyplot as plt
import scipy.io as scio

import h5py
import scipy.stats as stats


def load_data_and_topology(data_dir, ratio):
    # 获取各种短线的index
    dataFile = '../N-1/AC118/data_n-1_118.mat'

    data = h5py.File(data_dir, 'r')
    x = list(data.keys())
    # data = scio.loadmat(dataFile)
    dataX = data['conved_Y_load'].value.T
    dataY = data['gotten_VA'].value.T
    dataY2 = data['gotten_VM'].value.T

    # 获取电导和电纳,
    G = dataX[:, :119*119]
    G = np.reshape(G, [dataX.shape[0], 119, 119])
    B = dataX[:, 119*119:119*119*2]
    B = np.reshape(B, [dataX.shape[0], 119, 119])
    structure = np.concatenate([G[:, :, :, np.newaxis], B[:, :, :, np.newaxis]], axis=3)
    dataX = dataX[:, 119*119*2:]
    dataY = np.concatenate([dataY2, dataY], axis=1)
    trainData = []
    trainLabel = []
    trainA = []
    for x in range(int(189*ratio)):
        for index in range(x, 20000, 189):
            trainData.append(dataX[index, :])
            trainLabel.append(dataY[index, :])
            trainA.append(structure[index, :, :, :])

    testData = []
    testLabel = []
    testA = []
    for x in range(int(189 * ratio), 189, 1):
        for index in range(x, 20000, 189):
            testData.append(dataX[index, :])
            testLabel.append(dataY[index, :])
            testA.append(structure[index, :, :, :])

    return np.array(trainData), np.array(trainLabel), np.array(trainA), \
           np.array(testData), np.array(testLabel), np.array(testA)

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
    structure = branch['Ybus_10']
    D = np.diag(np.sum(structure, axis=0)**(-0.5))
    structure = np.dot(np.dot(D, structure), D)
    print(structure)
    print(np.real(structure))
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


if __name__=="__main__":
    res = load_data_and_topology("../N-1/AC118/data_n-1_118.mat", 0.9)