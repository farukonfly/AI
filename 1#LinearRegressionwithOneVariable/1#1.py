import tensorflow as tf

tf.reset_default_graph()  # 清除default graph和不断增加的节点

# 定义变量 a
a = tf.Variable(1, name='a')
# 定义操作b为a+1
b = tf.add(a, 1, name='b')
# 定义操作c为b*4
c = tf.multiply(b, 4, name='c')
# 定义操作d为c-b
d = tf.subtract(c, b, name='d')

logdir = "D:/log"

# 生成一个写日志的writer,将当前的TensorFlow计算图写入日志
writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
writer.close()
