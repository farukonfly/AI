import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# (1/6)人工数据集生成
np.random.seed(5)  # 设置随机数种子
x_data = np.linspace(-1, 1, 100)  # 等差数列,-1~1之间,100个

# y=2x+1+噪声,其中噪声的维度与x_data一致
y_data = 2*x_data+1.0+np.random.randn(*x_data.shape)*0.4

# 画出生成的图像
# plt.scatter(x_data, y_data)
# 画出想要学习的线性函数y=2x+1
# plt.plot(x_data, 2 * x_data + 1.0, color='red', linewidth=3)
# plt.show()

# 定义训练数据的占位符,x是特征值,y是标签值
x = tf.placeholder("float", name="x")
y = tf.placeholder("float", name="y")


def model(x, w, b):
    """(2/6)模型函数"""
    return tf.multiply(x, w) + b


# 定义变量
w = tf.Variable(1.0, name="w0")
# 定义变量
b = tf.Variable(0.0, name="b0")

# pred 预测值,前向计算
pred = model(x, w, b)

# 训练参数
# 迭代次数(训练轮数)
train_epochs = 10
# 学习率
learning_rate = 0.05

# (3/6)损失函数
# 均方差损失函数
loss_function = tf.reduce_mean(tf.square(y - pred))

# (4/6)优化器
# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(loss_function)


# 声明会话
sess = tf.Session()

# 在真正进行计算之前,需要将所有变量初始化
# 通过tf.global_variables_initializer函数可实现对所有变量的初始化
init = tf.global_variables_initializer()

sess.run(init)

# (5/6)迭代训练
# 训练中显示损失值
step = 0
loss_list = []
display_step = 10
for epoch in range(train_epochs):
    for xs, ys in zip(x_data, y_data):
        _, loss = sess.run([optimizer, loss_function],
                           feed_dict={x: xs, y: ys})
        loss_list.append(loss)
        step = step + 1
        if step % display_step == 0:
            print("Train Epoch:", '%02d' % (epoch+1), "Step %03d" %
                  (step), "loss=", "{:.9f}".format(loss))
    b0temp = b.eval(session=sess)
    w0temp = b.eval(session=sess)
#   plt.plot(x_data, w0temp*x_data+b0temp)


print("w:", sess.run(w))
print("b:", sess.run(b))

# plt.scatter(x_data, y_data, label="Original data")
# plt.plot(x_data, x_data*sess.run(w)+sess.run(b),
#         label="Fitted line", color="r", linewidth=3)
# plt.legend(loc=2)  # 指定图列位置
plt.plot(loss_list)
plt.show()

# (6/6)模型预测
x_test = 3.21
predict = sess.run(pred, feed_dict={x: x_test})
target = 2*x_test+1.0
print("预测值:%f" % predict)
print("目标值:%f" % target)
