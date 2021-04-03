---
layout: post
title: "NN for MNIST(2) 성능 향상시키기"
---

# NN for MNIST(2) 성능 향상시키기

## 성능을 향상시키기 위한 방법.

1. Sigmiod가 아닌 ReLU 사용하기.  
Sigmiod는 결과값의 범위가 0~1이기 때문에 layer가 깊어지면 제대로 작동되지 않는 문제가 있다.  
ReLu는 입력값이 0보다 작으면 0이고 0보다 크면 결과값이 비례해서 계속 커지는 함수이다.
2. weigt 초기값 조정하기  
Xavier initialization: 입력값의 갯수와 출력값의 갯수를 이용해서 초기값을 설정한다.  
3. Dropout 사용하기  
랜덤하게 일부 뉴런들의 연결을 끊어버려서 Overfitting을 방지한다. 학습할때만 0.5~0.7의 비율 사용하고 실전에는 사용하면 안된다.



```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

tf.set_random_seed(777)

# mnist 데이터 로드
mnist= input_data.read_data_sets("MNIST_data/", one_hot=True)
# xavier 초기화 사용시 변수가 이미 존재한다는 오류가 떠서 추가.(그래프 초기화)
tf.reset_default_graph()
#parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int (mnist.train.num_examples / batch_size)
# input
X = tf.placeholder(tf.float32,[None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# dropout (keep_prob) rate 0.7
keep_prob = tf.placeholder(tf.float32)
# xavier 초기화 사용
# relu를 사용하여 layer 설정.
# dropout 사용
W1 = tf.get_variable("W1", shape=[784, 512],initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[512, 512],initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[512, 512],initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[512, 512],initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[512, 10],initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4, W5) + b5

# cost/loss & optimizer 정의
cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train
for epoch in range(training_epochs):
    avg_cost = 0
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
        c, _ = sess.run([cost,optimizer], feed_dict= feed_dict)
        avg_cost += c / total_batch
    
    print('Epoch: ', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))
print('Learning Finished!')

#accuracy 체크하기.
correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy: ', sess.run(accuracy, feed_dict={ X: mnist.test.images, Y: mnist.test.labels,keep_prob:1}))

# 무작위 test 결과 출력해보기.
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))

plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
    Epoch:  0001 cost =  0.453380440
    Epoch:  0002 cost =  0.171895763
    Epoch:  0003 cost =  0.134768132
    Epoch:  0004 cost =  0.110966461
    Epoch:  0005 cost =  0.096886198
    Epoch:  0006 cost =  0.086131740
    Epoch:  0007 cost =  0.074485826
    Epoch:  0008 cost =  0.070590070
    Epoch:  0009 cost =  0.067031595
    Epoch:  0010 cost =  0.063140516
    Epoch:  0011 cost =  0.057054896
    Epoch:  0012 cost =  0.055115012
    Epoch:  0013 cost =  0.051370878
    Epoch:  0014 cost =  0.047782788
    Epoch:  0015 cost =  0.045051503
    Learning Finished!
    Accuracy:  0.9812
    Label:  [9]
    Prediction:  [9]
    


    
![png](/assets/images/21-04-03-NNforMNIST(2)/output_1_1.png)
    

