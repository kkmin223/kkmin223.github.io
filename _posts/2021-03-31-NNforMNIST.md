---
layout: post
title: "Wide and Deep NN for MNIST"
---

# Wide and Deep NN for MNIST



```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

mnist= input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10
#MNIST data image of shape 28*28 = 784
X= tf.placeholder(tf.float32,[None,784])
# 0 - 9 digits recognition = 10 classes
Y=tf.placeholder(tf.float32,[None,nb_classes])
#layer 만들기 (using softmax)
W1= tf.Variable(tf.random_normal([784,900]))
b1= tf.Variable(tf.random_normal([900]))
layer1 = tf.nn.softmax(tf.matmul(X,W1)+b1)

W2= tf.Variable(tf.random_normal([900,900]))
b2= tf.Variable(tf.random_normal([900]))
layer2 = tf.nn.softmax(tf.matmul(layer1,W2)+b2)


W3= tf.Variable(tf.random_normal([900,nb_classes]))
b3= tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.nn.softmax(tf.matmul(layer2,W3)+b3)



cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

#Test moder
is_correct= tf.equal(tf.arg_max(hypothesis,1),tf.arg_max(Y,1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

#parameters
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost=0
        total_batch = int(mnist.train.num_examples / batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost,optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch
        
        print('Epoch:', '%04d' %(epoch+1), 'cost=', '{:.9f}', format(avg_cost))
        # Test the model using test sets
        print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
        #Sample image sho and prediction
        #Get one and predict
        r=random.randint(0,mnist.test.num_examples-1)
        print("Lable:",sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
        print("Prediction:", sess.run(tf.argmax(hypothesis,1),feed_dict={X: mnist.test.images[r:r+1]}))
        plt.imshow(mnist.test.images[r:r+1].reshape(28,28),cmap='Greys', interpolation='nearest')
        plt.show()
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
    Epoch: 0001 cost= {:.9f} 2.323648843765259
    Accuracy:  0.1357
    Lable: [6]
    Prediction: [1]
    


    
![png](assets/images/2021-03-31-NNforMNIST/output_1_1.png)
    


    Epoch: 0002 cost= {:.9f} 2.2935774317654696
    Accuracy:  0.1386
    Lable: [5]
    Prediction: [1]
    


    
![png](assets/images/2021-03-31-NNforMNIST/output_1_3.png)
    


    Epoch: 0003 cost= {:.9f} 2.2892594268105233
    Accuracy:  0.1626
    Lable: [4]
    Prediction: [1]
    


    
![png](assets/images/2021-03-31-NNforMNIST/output_1_5.png)
    


    Epoch: 0004 cost= {:.9f} 2.2841259470852933
    Accuracy:  0.1852
    Lable: [2]
    Prediction: [1]
    


    
![png](assets/images/2021-03-31-NNforMNIST/output_1_7.png)
    


    Epoch: 0005 cost= {:.9f} 2.277156115445225
    Accuracy:  0.2113
    Lable: [2]
    Prediction: [0]
    


    
![png](assets/images/2021-03-31-NNforMNIST/output_1_9.png)
    


    Epoch: 0006 cost= {:.9f} 2.261625883362508
    Accuracy:  0.2282
    Lable: [9]
    Prediction: [0]
    


    
![png](assets/images/2021-03-31-NNforMNIST/output_1_11.png)
    


    Epoch: 0007 cost= {:.9f} 2.2159577699141066
    Accuracy:  0.2595
    Lable: [3]
    Prediction: [3]
    


    
![png](assets/images/2021-03-31-NNforMNIST/output_1_13.png)
    


    Epoch: 0008 cost= {:.9f} 2.1609244736758138
    Accuracy:  0.2934
    Lable: [6]
    Prediction: [1]
    


    
![png](assets/images/2021-03-31-NNforMNIST/output_1_15.png)
    


    Epoch: 0009 cost= {:.9f} 2.095529578815809
    Accuracy:  0.3288
    Lable: [5]
    Prediction: [3]
    


    
![png](assets/images/2021-03-31-NNforMNIST/output_1_17.png)
    


    Epoch: 0010 cost= {:.9f} 2.001651955301111
    Accuracy:  0.3551
    Lable: [4]
    Prediction: [7]
    


    
![png](assets/images/2021-03-31-NNforMNIST/output_1_19.png)
    


    Epoch: 0011 cost= {:.9f} 1.8807480469616997
    Accuracy:  0.4005
    Lable: [8]
    Prediction: [3]
    


    
![png](assets/images/2021-03-31-NNforMNIST/output_1_21.png)
    


    Epoch: 0012 cost= {:.9f} 1.763005409240722
    Accuracy:  0.4471
    Lable: [4]
    Prediction: [7]
    


    
![png](assets/images/2021-03-31-NNforMNIST/output_1_23.png)
    


    Epoch: 0013 cost= {:.9f} 1.6536300700361075
    Accuracy:  0.475
    Lable: [0]
    Prediction: [0]
    


    
![png](assets/images/2021-03-31-NNforMNIST/output_1_25.png)
    


    Epoch: 0014 cost= {:.9f} 1.566138184287331
    Accuracy:  0.5002
    Lable: [0]
    Prediction: [0]
    


    
![png](assets/images/2021-03-31-NNforMNIST/output_1_27.png)
    


    Epoch: 0015 cost= {:.9f} 1.5014359296451922
    Accuracy:  0.5462
    Lable: [8]
    Prediction: [5]
    


    
![png](assets/images/2021-03-31-NNforMNIST/output_1_29.png)
    


## layer를 추가해 보고 느낀점.
layer를 4개까지 추가해 봤지만 오히려 accuracy가 0.11까지 떨어지면서 전혀 학습이 되지않았다 그래서 layer를 2개로 줄였지만 여전히 1개였을때 보다 더 떨어지는 accuracy가 나타났다 이번엔 layer를 3개로 늘리고 더 wide하게 해봤지만 더 성능이 떨어지는 것이 확인 되었다.    
무작정 wide and deep layer를 쓴다고 더 좋은게 아니라는 것을 알게되었다.
