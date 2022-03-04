import os
from PIL import Image
import numpy as np
import h5py
import matplotlib.pyplot as plt

# 数据预处理,训练数据：输入33*33，输出21*21
def deal_data(scale,stride,x_size,label_size,dir='test'):
    if dir == 'test':
        os.chdir('F:\\model\SRCNN-Tensorflow-master\\SRCNN-Tensorflow-master\\Test\\Set5')
    elif dir == 'train':
        os.chdir('F:\\model\\SRCNN-Tensorflow-master\\SRCNN-Tensorflow-master\\Train')
    file_names = os.listdir()
    data = []    # 用于储存处理好的输入数据
    label = []   # 用于存储处理好的输出数据
    padding = int((x_size - label_size) / 2)    # 使用的输入为33*33 标签为该图中心点周围21*21，因此要计算padding用于找寻中心点
    for i in range(len(file_names)):
        img = Image.open(file_names[i])
        img_array = np.array(img)

        # 将图片裁剪到能被scale整除，因为要模拟lr放到scale倍到hr
        h,w,_ = img_array.shape
        h = h - np.mod(h,scale)   # scale为图片放大倍数
        w = w - np.mod(w,scale)
        # label_为y，x_为输入
        label_ = img_array[0:h,0:w,:]
        print('正在处理第{}张图像：{} * {}'.format(i+1,h,w))
        # 使用两次bicubic插值，先缩小scale倍，再放大scal倍，因为SRCNN放大图片的过程为先使用bicubic将lr图像放大至和sr相同分辨率，之后再使用网络超分
        label_img = Image.fromarray(label_)
        x_img = label_img.resize((int(w / scale),int(h / scale)),Image.BICUBIC)
        x_img = x_img.resize((int(w),int(h)),Image.BICUBIC)
        x_ = np.array(x_img)

        # 转换为ycbcr，并取y通道作为输入
        label_ = rgb2ycbcr(label_)
        x_ = rgb2ycbcr(x_)
        # 图片分割
        FG_data(x_,label_,x_size,label_size,padding,stride,data,label)

    # 将处理好的数据保存为h5文件
    data = np.array(data)
    label = np.array(label)
    print('data',data.shape)
    print('label',label.shape)
    if dir == 'test':
        savepath = 'F:\\model\\SRCNN-Tensorflow-master\\SRCNN-Tensorflow-master\\FuXianSRCNN\\test\\test.h5'
    elif dir == 'train':
        savepath = 'F:\\model\\SRCNN-Tensorflow-master\\SRCNN-Tensorflow-master\\FuXianSRCNN\\train\\train.h5'

    with h5py.File(savepath,'w') as hf:
        hf.create_dataset('data',data=data)
        hf.create_dataset('label',data=label)


# rgb2ycbcr
def rgb2ycbcr(x):
    # SRCNN是对Y通道进行超分,因此，此函数只返回y通道
    y = 0.299 * x[:,:,0] + 0.587 * x[:,:,1] + 0.114 * x[:,:,2]
    return y

# 数据切割
def FG_data(x_,label_,x_size,label_size,padding,stride,data,label):
    h,w = x_.shape
    for i in range(0,h - x_size + 1,stride):
        for j in range(0,w - x_size + 1,stride):
            x_i = x_[i:i + x_size,j:j + x_size]
            x_i = np.reshape(x_i,[x_size,x_size,1])
            data.append(x_i)
            label_i = label_[i + padding:i + padding + label_size,j + padding:j + padding + label_size]
            label_i = np.reshape(label_i,[label_size,label_size,1])
            label.append(label_i)

# 加载数据
def load_data():
    os.chdir('F:\\model\\SRCNN-Tensorflow-master\\SRCNN-Tensorflow-master\\FuXianSRCNN\\train')
    train_data = h5py.File('train.h5','r')
    # 传入神经网络的x要进行归一化
    train_x = np.array(train_data['data']) / 255
    train_y = np.array(train_data['label']) / 255
    os.chdir('F:\\model\\SRCNN-Tensorflow-master\\SRCNN-Tensorflow-master\\FuXianSRCNN\\test')
    test_data = h5py.File('test.h5','r')
    test_x = np.array(test_data['data']) / 255
    test_y = np.array(test_data['label']) / 255

    return train_x,train_y,test_x,test_y

def data_show(data,num):
    n = int(np.sqrt(num))
    fig,ax = plt.subplots(ncols=n,nrows=n,sharey=True,sharex=True)
    for i in range(n):
        for j in range(n):
            img_i = np.array(data[i * n + j,:,:])
            ax[i,j].imshow(img_i)
            plt.yticks(np.array([]))
            plt.xticks(np.array([]))
    plt.show()


if __name__ == '__main__':
    scale = 3
    stride = 14
    x_size = 33
    label_size = 21
    deal_data(scale,stride,x_size,label_size)
    # train_x, train_y, test_x, test_y = load_data()
    # data_show(train_x,25)
    # data_show(train_y,25)



