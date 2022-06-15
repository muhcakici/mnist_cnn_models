import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1)
x_train = np.repeat(x_train, 3, axis=-1)
x_train = x_train.astype('float32') / 255
x_train = tf.image.resize(x_train, [32,32]) # if we want to resize
y_train = tf.keras.utils.to_categorical(y_train , num_classes=10)
print(x_train.shape, y_train.shape)

input = tf.keras.Input(shape=(32,32,3))
efnet = tf.keras.applications.ResNet50(weights='imagenet',include_top = False,input_tensor = input)
gap = tf.keras.layers.GlobalMaxPooling2D()(efnet.output)
output = tf.keras.layers.Dense(10, activation='softmax', use_bias=True)(gap)
func_model = tf.keras.Model(efnet.input, output)
func_model.compile(
          loss  = tf.keras.losses.CategoricalCrossentropy(),
          metrics = tf.keras.metrics.CategoricalAccuracy(),
          optimizer = tf.keras.optimizers.Adam())

func_model.fit(x_train, y_train, batch_size=128, epochs=5, verbose = 2)
model.evaluate(x_test, y_test)