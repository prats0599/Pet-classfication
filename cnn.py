# Convolutional Neural Networks
# 1.Convolution(with activation fn to achieve non-linearity)-->2.Max pooling --> 3.Flattening-->4.Full connection

# building the cnn

# Importing keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D

# Initializing the CNN
classifier = Sequential()

# step1: Convolution
classifier.add(Convolution2D(filters = 32, kernel_size = 3, input_shape = (64, 64, 3), activation = 'relu'))

# step2: Max Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# added layer -->loss: 0.2045 - accuracy: 0.9185 - val_loss: 1.2913 - val_accuracy: 0.7965
# normal -->loss: 0.3903 - accuracy: 0.8234 - val_loss: 0.1769 - val_accuracy: 0.7830
# compared to training acc. hence second layer is added as common practice.
# You can also increase the filters to 64 on the added convolutional layer.
# Adding a second convolutional layer to improve accuracy
# could also choose higher target size for the images. But time consumption will increase.
#classifier.add(Convolution2D(filters = 32, kernel_size = 3, activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))


# step3: Flattening
classifier.add(Flatten())

# step4: Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling cnn
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the cnn to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,  # all pixels will be between 0 and 1
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                              target_size=(64, 64),
                                              batch_size=32,
                                              class_mode='binary')
	
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

bs = 32
classifier.fit_generator(training_set,
                         steps_per_epoch=8000/bs,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000/bs)
