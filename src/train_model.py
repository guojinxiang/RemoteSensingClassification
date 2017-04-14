from keras.preprocessing.image import ImageDataGenerator

from cnn import CNN

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)
# 记得一定要以灰度图的方式读入
train_generator = train_datagen.flow_from_directory('../resources/data/train', target_size=(25, 25),
                                                    color_mode='grayscale',
                                                    class_mode='binary')
validation_generator = test_datagen.flow_from_directory('../resources/data/validation', target_size=(25, 25),
                                                        color_mode='grayscale',
                                                        class_mode='binary')
# print(train_generator.image_shape)
model = CNN()
model.fit_generator(train_generator, steps_per_epoch=100, epochs=50, validation_data=validation_generator,
                    validation_steps=100)
model.save_weights('../resources/cnn.h5')
