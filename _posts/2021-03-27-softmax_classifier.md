---
layout : post
title: "Softmax Classifier"
---

# Softmax Classifier 구현

### Softmax function
softamx는 input에 대한 여러 개의 연산 결과를 정규화하여 모든 결과값의 합을 1로 만들어 주는것이다. 또 이러한 값들을 one-hot encoding을 이용하여 정답 레이블을 학습시킨다,
#### one-hot encoding
one-hot encoding은 분류 문제가 발생한 경우 대상 변수 또는 클래스를 나타내는 방법이다. 대상 변수 또는 클래스만 1로 표현하고, 나머지는 모두 0으로 표현한다.




```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], 
                                                        [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

X=tf.placeholder("float",[None,4])
Y=tf.placeholder("float",[None,3])
nb_classes = 3

W=tf.Variable(tf.random_normal([4, nb_classes]),name='weight')
b=tf.Variable(tf.random_normal([nb_classes]),name='bias')

# softmax= exp(Logits) / reduce_sum(exp(Logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)

#Cross entropy cost/loss
cost=tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
print("---Training---")
for step in range(2001):
    sess.run(optimizer,feed_dict={X: x_data, Y: y_data})
    if step%200 == 0:
        print(step,sess.run(cost,feed_dict={X:x_data,Y: y_data}))

# Test & one-hot encoding
print("\n---Test---")
all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9], 
                                          [1, 3, 4, 3], 
                                          [1, 1, 0, 1]]})
print(all, sess.run(tf.arg_max(all, 1)))

```

    ---Training---
    0 7.5801344
    200 0.6259205
    400 0.5234496
    600 0.4333104
    800 0.34377247
    1000 0.25572217
    1200 0.21790013
    1400 0.19866323
    1600 0.18243599
    1800 0.16857514
    2000 0.15660769
    
    ---Test---
    [[4.1268948e-03 9.9586159e-01 1.1524327e-05]
     [8.7300026e-01 1.0752687e-01 1.9472882e-02]
     [1.8701536e-08 3.5809542e-04 9.9964190e-01]] [1 0 2]
    


```python

```
