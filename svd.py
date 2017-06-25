
import tensorflow as tf
import numpy as np


k = 2
A = [[1.0, 2], [3.0, 4], [5.0, 6], [7.0, 8]]
row = len(A)
col = len(A[0])
A = tf.convert_to_tensor(A, dtype='float32')
s = A.get_shape()
E_U = tf.convert_to_tensor(np.eye(k), dtype='float32')
E_V = tf.convert_to_tensor(np.eye(k), dtype='float32')




U = tf.Variable(tf.truncated_normal([row, k], stddev=0.1), dtype='float32')
V = tf.Variable(tf.truncated_normal([col, k], stddev=0.1), dtype='float32')
S = tf.Variable(tf.truncated_normal([1, k], stddev=0.1), dtype='float32')



alpha = 5
loss = tf.nn.l2_loss(A - tf.matmul(U*S, tf.transpose(V))) + alpha * (tf.nn.l2_loss(tf.matmul(tf.transpose(U), U) - E_U) + tf.nn.l2_loss(tf.matmul(tf.transpose(V), V) - E_V))
train_step = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


for i in range(100000):
    sess.run(train_step)
    if i % 100 == 0:
        l_loss = sess.run(loss)
        print("loss:", l_loss)
        if l_loss < 0.000001:
            break

print("E_U:", sess.run(E_U))
print("E_V:", sess.run(E_V))
print("U:", sess.run(U))
print("S:", sess.run(S))
print("V:", sess.run(V))
print("UUt:", sess.run(tf.matmul(tf.transpose(U), U)))
print("VVt:", sess.run(tf.matmul(V, tf.transpose(V))))








