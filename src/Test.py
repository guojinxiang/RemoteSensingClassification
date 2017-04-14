import numpy as np
from PIL import Image

from cnn import CNN

# 载入模型
model = CNN()
model.load_weights('../resources/cnn.h5')

test_img = Image.open('../resources/testimg.jpg')
# test_img.show()
# print(test_img.size)

# 遍历图片，截取出25*25的测试图放入神经网络中进行分类判别
test_region = []
for i in range(test_img.size[0]):
    for j in range(test_img.size[1]):
        # 要先将图片像素值缩放到0-1之间
        region = (np.asarray(test_img.crop(box=(i, j, i + 25, j + 25))) / 255)
        # print(region.shape)
        test_region.append(region)
model.predict(test_region, verbose=1)
