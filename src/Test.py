from PIL import Image

from cnn import CNN

test_img = Image.open('../resources/testimg.jpg')
print(test_img)

model = CNN()
model.load_weights('../resources/cnn.h5')

# model.predict(test_img,verbose=1)
