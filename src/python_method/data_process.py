import numpy as np
import scipy.io as sio
from PIL import Image

train_set_1 = sio.loadmat('../../resources/training_instance_matrix.mat')['training_instance_matrix']
label_set_1 = sio.loadmat('../../resources/training_label_vector.mat')['training_label_vector']
train_set_2 = sio.loadmat('../../resources/training_instance_matrix2.mat')['training_instance_matrix2']
label_set_2 = sio.loadmat('../../resources/training_label_vector2.mat')['training_label_vector2']
shape = (25, 25)

data_set = np.concatenate((train_set_1, train_set_2))
labels = np.concatenate((label_set_1, label_set_2))

for i in range(len(data_set)):
    # 一定要保存为灰度图像，因为训练数据集本身就是灰度图
    img = Image.fromarray(data_set[i].reshape(shape), mode='L')
    # print(img.mode,img.size)
    if labels[i] == 0:
        img.save('../../resources/data/train/negative/' + str(i) + '.jpg')
    else:
        img.save('../../resources/data/train/positive/' + str(i) + '.jpg')
