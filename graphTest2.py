import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import scipy.io as scio


def conv_layer(structure, layers, n_units, name, activate=True):
    outputs = []
    for num in range(n_units):
        w0 = np.random.normal(0, 0.5, size=(structure.shape[1], structure.shape[2]))
        b0 = np.random.normal(0, 0.5, size=structure.shape[1])
        with tf.name_scope(name):
            w = tf.Variable(w0, name='w', dtype='float32', trainable=True)
            w = tf.multiply(structure, w)
            b = tf.Variable(b0, name='b', dtype='float32', trainable=True)
            output = tf.reduce_sum(tf.matmul(w, layers), axis=-1) + b
            outputs.append(output)
    outputs = tf.stack(outputs, axis=-1)
    if activate:
        return tf.nn.relu(outputs)
    else:
        return outputs


def minibatcher(trainData, trainLabel, trainA, batch_size, shuffle):
    idx = np.random.permutation(trainData.shape[0])
    for k in range(int(np.ceil(trainData.shape[0] / batch_size))):
        from_idx = k * batch_size
        to_idx = (k + 1) * batch_size
        trainx = trainData[idx[from_idx:to_idx], :]
        trainy = trainLabel[idx[from_idx:to_idx], :]
        traina = trainA[idx[from_idx:to_idx], :, :]

        trainx = np.concatenate([trainx[:, :14, np.newaxis], trainx[:, 14:, np.newaxis]], axis=2)
        trainy = np.concatenate([trainy[:, :14, np.newaxis], trainy[:, 14:, np.newaxis]], axis=2)
        yield trainx, trainy, traina


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def model(structure, inputPq):
    l1 = conv_layer(structure, inputPq, n_units=4, name='l1')
    l2 = conv_layer(structure, l1, n_units=8, name='l2')
    l3 = conv_layer(structure, l2, n_units=16, name='l3')
    l4 = conv_layer(structure, l3, n_units=16, name='l4')
    l5 = conv_layer(structure, l4, n_units=8, name='l5')
    l6 = conv_layer(structure, l5, n_units=4, name='l6', activate=True)
    l7 = conv_layer(structure, l6, n_units=2, name='l7', activate=False)
    return l7


if __name__ == "__main__":
    from graphLayer import load_data_AC118_two_distribution, Z_Score

    # 准备训练集
    trainData, trainLabel, trainA, testData, testLabel, testA = load_data_AC118_two_distribution(0.9)

    # 对数据进行归一化处理
    trainData, mu1, sgma1 = Z_Score(trainData)
    trainLabel, mu, sgma = Z_Score(trainLabel)

    batch_size = 100
    max_epochs = 800
    lr = 0.001
    save_pred_every = 1
    savedir = './case14/'

    inputPq = tf.compat.v1.placeholder(tf.float32, shape=(None, 14, 2))
    structure = tf.compat.v1.placeholder(tf.float32, shape=(None, 14, 14))
    label_v = tf.compat.v1.placeholder(tf.float32, shape=(None, 14, 2))

    #  model
    layer_v = model(structure, inputPq)

    # 计算损失
    deta = layer_v - label_v
    angle, mag = tf.unstack(deta, axis=-1)
    lossMag = tf.reduce_mean(tf.math.square(mag))
    lossMag_var = tf.math.reduce_variance(mag)

    lossAngle = tf.reduce_mean(tf.math.square(angle))
    lossAngle_var = tf.math.reduce_variance(angle)
    loss = lossMag*0.4 + lossAngle*0.6
    tf.summary.scalar("Total loss", loss)
    tf.summary.scalar("Magnitude loss", lossMag)
    tf.summary.scalar("Magnitude var loss", lossMag_var)
    tf.summary.scalar('Angle loss', lossAngle)
    tf.summary.scalar('Angle var loss', lossAngle_var)

    with tf.name_scope("Train"):
        # some optimisation strategy, arbitrary learning rate
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.compat.v1.train.exponential_decay(lr,
                                                             global_step,
                                                             2000, 0.96, staircase=True)
        tf.summary.scalar('LearningRate', learning_rate)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate, name="optimizer_adam")
        train_op = optimizer.minimize(loss, global_step=global_step, name="train_op")

    with tf.compat.v1.Session() as sess:
        restore_var = tf.global_variables()
        sess.run(tf.compat.v1.global_variables_initializer())
        summary_merge = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir=savedir+"log", graph=sess.graph)
        ckpt = tf.train.get_checkpoint_state(savedir+'model/')
        if ckpt and ckpt.model_checkpoint_path:
            loader = tf.train.Saver(var_list=restore_var)
            # load(loader, sess, './case118/model/model.ckpt-2901360')
            load(loader, sess, ckpt.model_checkpoint_path)
        # inputs = np.random.rand(1, 4)
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
        for epoch in range(max_epochs):
            print("Epoch=", epoch)
            tf_score = []
            for mb in tqdm(minibatcher(trainData, trainLabel, trainA, batch_size=batch_size, shuffle=True)):
                tf_output = sess.run([train_op, loss, summary_merge, global_step, learning_rate, layer_v],
                                        feed_dict={inputPq: mb[0],
                                                   label_v: mb[1],
                                                   structure: mb[2]})
                tf_score.append(tf_output[1])
                writer.add_summary(tf_output[2], tf_output[3])
            print(" train_loss_score=", np.mean(tf_score), "Lr=", tf_output[4])
            if epoch % save_pred_every == 0:
                save(saver, sess, savedir+'model', step=epoch)






