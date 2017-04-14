import numpy as np
from PIL import Image

from cnn import CNN

# 载入模型
model = CNN()
model.load_weights('../resources/cnn.h5')

# 转为灰度图
test_img = Image.open('../resources/testimg.jpg').convert('L')
# test_img.show()
# print(test_img.size,test_img.mode)
# print(test_img.crop(box=(0, 0, 25, 25)).mode,test_img.crop(box=(0, 0, 25, 25)).size)
# region = (np.asarray(test_img.crop(box=(0, 0, 25, 25))) / 255).reshape((25,25,1))
# print(region.shape,region.size)

# 遍历图片，截取出25*25的测试图放入神经网络中进行分类判别
test_region = []
for i in range(test_img.size[0]):
    for j in range(test_img.size[1]):
        # 要先将图片像素值缩放到0-1之间，并且reshape,因为PIL读取的图片默认shape是没有第三维的
        region = (np.asarray(test_img.crop(box=(i, j, i + 25, j + 25))) / 255).reshape((25, 25, 1))
        test_region.append(region)
model.predict(test_region, verbose=1)
