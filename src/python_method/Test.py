import math

import numpy as np
from PIL import Image, ImageDraw

from ..python_method.cnn import CNN
from ..python_method.non_max_suppression import non_max_suppression

# 载入模型
model = CNN()
model.load_weights('../../resources/cnn.h5')

# 转为灰度图
test_img = Image.open('../../resources/testimg.jpg').convert('L')
# test_img.show()
# print(test_img.size,test_img.mode)
# print(test_img.crop(box=(0, 0, 25, 25)).mode,test_img.crop(box=(0, 0, 25, 25)).size)
# region = (np.asarray(test_img.crop(box=(0, 0, 25, 25))) / 255).reshape((25,25,1))
# print(region.shape,region.size)

boxs = []
scale = 1
# 构建图像金子塔，用来做多尺度分割(非常耗时!)
while test_img.size[0] >= 25 and test_img.size[1] >= 25:
    # 遍历图片，截取出25*25的测试图放入神经网络中进行分类判别
    for i in range(test_img.size[0] - 25):
        for j in range(test_img.size[1] - 25):
            # 要先将图片像素值缩放到0-1之间，并且reshape,因为PIL读取的图片默认shape是没有第三维的，
            # 所以需要最后一个值设为1，表示一个通道，即灰度值，又因为模型的预测函数需要输入一个
            # 预测列表，现在我需要对每一个单独处理，所以在第一个值上设为1，表示这批数据只有一个待预测数据
            region = (np.asarray(test_img.crop((i, j, i + 25, j + 25))) / 255).reshape((1, 25, 25, 1))
            # print(region)
            if model.predict_classes(region, verbose=0)[0][0] == 1:
                # 这里一定要注意乘以尺度，因为图像经过缩放，此时的坐标是原图的1/scale倍，所以需要还原回去,
                # 同时需要注意向上取整，因为图像缩放的时候向下取整了
                boxs.append([math.ceil(i * scale), math.ceil(j * scale), math.ceil((i + 25) * scale),
                             math.ceil((j + 25) * scale)])
                # print(boxs)
    # scale用来控制图像缩放的尺度，每次缩小为原先的1/1.1倍
    scale += 0.1
    # 注意要向下取整，因为resize的参数必须是整数
    test_img = test_img.resize((int(test_img.size[0] / scale), int(test_img.size[1] / scale)))

# 需要转成numpy数组，因为非极大值抑制需要传入numpy数组
boxs = np.array(boxs)
print(str(len(boxs)) + ' boxs extracted')

# 进行非极大值抑制,认为两个box达到10%的覆盖就可以执行抑制，只留下一个了
boxs = non_max_suppression(boxs, overlapThresh=0.1)
print(str(len(boxs)) + ' boxs kept')
img = Image.open('../../resources/testimg.jpg')
draw = ImageDraw.Draw(img)
for box in boxs:
    draw.rectangle([(box[0], box[1]), (box[2], box[3])])

# img.show()
img.save('../../resources/testimg_multi_scale_result.jpg')
print('prediction completed')
