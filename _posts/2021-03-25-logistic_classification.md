---
layout = post
title="Logistic Classification"
---


# Logistic (regression) classifier 구현



```python
import tensorflow as tf
import numpy as np

# x and y data
x=[[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y=[[0],[0],[0],[1],[1],[1]]

# sigmoid는 0에서 1 사이의 값이 나온다.
# Y 가 0 or 1인 분류문제이므로 사용한다.
# layer 만들기
sgd=tf.keras.optimizers.SGD(learning_rate=0.01)
model= tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1,activation="sigmoid",input_dim=2))
# 훈련 환경 설정
model.compile(loss='mean_squared_error',optimizer=sgd)
# 훈련 하기
model.fit(x,y,epochs=100)

print(model.evaluate(x,y))
```

    Epoch 1/100
    1/1 [==============================] - 0s 0s/step - loss: 0.4010
    Epoch 2/100
    1/1 [==============================] - 0s 0s/step - loss: 0.4004
    Epoch 3/100
    1/1 [==============================] - 0s 0s/step - loss: 0.3999
   
    Epoch 99/100
    1/1 [==============================] - 0s 0s/step - loss: 0.3188
    Epoch 100/100
    1/1 [==============================] - 0s 5ms/step - loss: 0.3175
    1/1 [==============================] - 0s 0s/step - loss: 0.3162
    0.316224604845047
    


```python

```
