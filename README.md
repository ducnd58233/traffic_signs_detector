## **Link dataset**
- Link dataset: [dataset](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

## **Model summary**
```
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 60, 60, 3)]       0

 conv2d (Conv2D)             (None, 58, 58, 32)        896

 conv2d_1 (Conv2D)           (None, 56, 56, 64)        18496

 max_pooling2d (MaxPooling2D  (None, 28, 28, 64)       0
 )

 batch_normalization (BatchN  (None, 28, 28, 64)       256
 ormalization)

 conv2d_2 (Conv2D)           (None, 26, 26, 128)       73856

 max_pooling2d_1 (MaxPooling  (None, 13, 13, 128)      0
 2D)

 batch_normalization_1 (Batc  (None, 13, 13, 128)      512
 hNormalization)

 global_average_pooling2d (G  (None, 128)              0
 lobalAveragePooling2D)

 dense (Dense)               (None, 128)               16512

 dense_1 (Dense)             (None, 43)                5547

=================================================================
Total params: 116,075
Trainable params: 115,691
Non-trainable params: 384
```

## **Evaluate model**
```
Evaluating validation set:
62/62 [==============================] - 2s 34ms/step - loss: 0.0615 - accuracy: 0.9832
Evaluating test set:
198/198 [==============================] - 7s 37ms/step - loss: 0.3682 - accuracy: 0.8966
```