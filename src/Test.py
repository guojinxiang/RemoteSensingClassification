from keras.preprocessing.image import ImageDataGenerator

from cnn import CNN

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory('../resources/data/train', target_size=(25, 25),
                                                    class_mode='binary')
validation_generator = test_datagen.flow_from_directory('../resources/data/validation', target_size=(25, 25),
                                                        class_mode='binary')

model = CNN(train_generator, validation_generator)
