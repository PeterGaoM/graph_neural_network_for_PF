import numpy as np
from numpy.linalg import inv, pinv
import matplotlib.pyplot as plt
import scipy.io as scio
import h5py
import scipy.stats as stats


def load_data_and_topology(data_dir, train_index, test_index):
    # 获取各种短线的index
    data = h5py.File(data_dir, 'r')
    x = list(data.keys())
    dataX = data['conved_Y_load'].value.T

    # 构建节点到支路的转移转移矩阵188*119，P用B转移，Q用G转移
    branch = data['branch'].value.T
    Y = data['Y'].value.T
    G = Y[:, :119 * 119]
    G = np.reshape(G, [Y.shape[0], 119, 119])
    B = Y[:, 119 * 119:119 * 119 * 2]
    B = np.reshape(B, [Y.shape[0], 119, 119])
    TransQ = np.zeros((dataX.shape[0], branch.shape[0], 119), dtype='float32')
    TransP = np.zeros((dataX.shape[0], branch.shape[0], 119), dtype='float32')
    for index in range(dataX.shape[0]):
        TransQ_temp = np.zeros((branch.shape[0], 119), dtype='float32')
        TransP_temp = np.zeros((branch.shape[0], 119), dtype='float32')
        for x in range(branch.shape[0]):
            if dataX[index, x] == 0.0: # 跳过断掉的线路
                continue
            TransP_temp[x, int(branch[x, 0])-1] = B[index, int(branch[x, 0])-1, int(branch[x, 1])-1]
            TransP_temp[x, int(branch[x, 1])-1] = B[index, int(branch[x, 0])-1, int(branch[x, 1])-1]
            TransQ_temp[x, int(branch[x, 0])-1] = G[index, int(branch[x, 0])-1, int(branch[x, 1])-1]
            TransQ_temp[x, int(branch[x, 1])-1] = G[index, int(branch[x, 0])-1, int(branch[x, 1])-1]
        TransQ_temp = TransQ_temp / (np.sum(TransQ_temp, axis=0) + 0.0001)  # 避免出现nan的归一化
        TransP_temp[TransP_temp >= 5000] = 0
        TransP_temp = TransP_temp / (np.sum(TransP_temp, axis=0) + 0.0001)  # 避免出现nan的归一化
        TransQ_temp[TransQ_temp >= 5000] = 0
        TransQ[index, :, :] = TransQ_temp
        TransP[index, :, :] = TransP_temp

    # 根据构建的负荷节点支路转移矩阵对负荷进行转移
    dataX = dataX[:, 188:]
    dataInput = np.zeros((dataX.shape[0], 188*2), dtype='float32')
    for index in range(dataX.shape[0]):
        P = dataX[index, :119]
        Q = dataX[index, 119:]
        dataInput[index, :188] = np.dot(TransP[index, :, :], P)
        dataInput[index, 188:] = np.dot(TransQ[index, :, :], Q)

    # 构建支路与支路的连接矩阵
    TransA = np.zeros((dataX.shape[0], branch.shape[0], branch.shape[0]), dtype='float32')
    for index in range(dataX.shape[0]):
        TransA_temp = np.zeros((branch.shape[0], branch.shape[0]), dtype='float32')
        for x in range(branch.shape[0]):
            if dataX[index, x] == 0.0:  # 跳过断掉的线路
                continue
            TransA_temp[int(branch[x, 0])-1, int(branch[x, 1])-1] = 1
            TransA_temp[int(branch[x, 1])-1, int(branch[x, 0])-1] = 1

        temp = np.sum(TransA_temp, axis=0)+1
        D = np.diag(temp ** (-0.5))
        TransA_temp = np.dot(np.dot(D, TransA_temp), D)
        TransA[index, :, :] = TransA_temp

    # 整理输出的数据
    dataY = data['gotten_PF'].value.T
    dataY2 = data['gotten_QF'].value.T
    dataY = np.concatenate([dataY2, dataY], axis=1)
    dataX = dataInput
    temp = np.sum(np.sum(G**2+B**2, axis=1), axis=1)

    trainData = []
    trainLabel = []
    trainA = []
    for x in train_index:
        res = np.where(temp == temp[x])[0]
        trainData.extend(dataX[res, :])
        trainLabel.extend(dataY[res, :])
        trainA.extend(TransA[res, :, :])

    testData = []
    testLabel = []
    testA = []
    for x in test_index:
        res = np.where(temp == temp[x])[0]
        testData.extend(dataX[res, :])
        testLabel.extend(dataY[res, :])
        testA.extend(TransA[res, :, :])

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


if __name__ == "__main__":
    res = load_data_and_topology("../N-1/AC118/data_n-1_118.mat", list(range(20)), list(range(20)))
    print(123)