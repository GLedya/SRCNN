import tensorflow as tf
from tensorflow.keras import Model
import MyUtils
import os
import matplotlib.pyplot as plt
from keras import backend as k
from tensorflow.keras import Input

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.c1 = tf.keras.layers.Conv2D(filters=64,kernel_size=9,strides=1,padding='valid',activation='relu')
        self.c2 = tf.keras.layers.Conv2D(filters=32,kernel_size=1,strides=1,padding='valid',activation='relu')
        self.c3 = tf.keras.layers.Conv2D(filters=1,kernel_size=5,strides=1,padding='valid')

    def call(self,x):
        a1 = self.c1(x)
        a2 = self.c2(a1)
        h = self.c3(a2)
        return h

def psnr(y_true,y_pred):
    return 10.0 * k.log(1.0 / (k.mean(k.square(y_pred - y_true)))) / k.log(10.0)

train_x,train_y,test_x,test_y = MyUtils.load_data()
print('train_x',train_x.shape)
print('train_y',train_y.shape)
print('test_x',test_x.shape)
print('test_y',test_y.shape)

model = MyModel()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss='mean_squared_error',metrics=[psnr])

model.build(input_shape=(None,33,33,1))
model.call(Input(shape=(33,33,1)))
print(model.summary())

os.chdir('F:\\model\\SRCNN-Tensorflow-master\\SRCNN-Tensorflow-master\\FuXianSRCNN\\checkpoint')
checkpoint_save_path = 'params.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('------导入模型继续训练------')
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(train_x,train_y,batch_size=128,epochs=15000,
                    shuffle = True,
                    validation_data=(test_x,test_y),validation_freq=1,
                    callbacks=[cp_callback])

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.subplot()
plt.plot(train_loss,label='train_loss')
plt.plot(val_loss,label='val_loss')
plt.legend()
plt.show()




