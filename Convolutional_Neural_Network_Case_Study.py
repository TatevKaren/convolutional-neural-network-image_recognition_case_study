import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

print(tf.__version__)

#--------------------- Data Preprocessing --------------------#

#feature training
train_datagen = ImageDataGenerator(
        # reducing/normalizing the pixels
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
#connecting the image augmentation tool to our dataset
train_set = train_datagen.flow_from_directory(
        'training_set',
        #final size of the images that will be fed into the ann
        target_size=(64, 64),
        # number of images that we want to have in each batch
        batch_size=32,
        # we have binary classification --> binary class mode
        class_mode='binary')


#only rescaling but no transformations
test_datagen = ImageDataGenerator(rescale=1./255)
#connecting to the test data
test_set = test_datagen.flow_from_directory(
        'test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

print(test_set)

#--------------------- Building CNN --------------------#
# initializing CNN as sequential layers
cnn = tf.keras.models.Sequential()

# Step 1: Convolution to get the Feature Map
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape=[64,64,3]))

# Step 2: Max Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2 ,strides=2))
#adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2 ,strides=2))

# Step 3: Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4: Full Connection
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))

# Step 5: Output Layer
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

#--------------------- Training the CNN --------------------#
#compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#training the CNN on the training set and evaluating it on the test set
cnn.fit(x = train_set, validation_data = test_set, epochs = 25)


#--------------------- Single prediction with CNN --------------------#
test_image = image.load_img('single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
# to convert image in pii format into a numpy array format
test_image = image.img_to_array(test_image)
# adding extra dimension to put this image into a batch by saying where we want to add this batch (as the first dimension)
test_image = np.expand_dims(test_image, axis = 0)
# cnn prediction on the test image
result = cnn.predict(test_image)
# getting the results encoding: which indices correspond to which classes (1: dog, 0:cat)
print(train_set.class_indices)

#prediction for the single image/element from the batch
if result[0][0] == 1:
   prediction = 'dog'
else:
   prediction = 'cat'

print(prediction)


test_image2 = image.load_img('single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
# to convert image in pii format into a numpy array format
test_image2 = image.img_to_array(test_image2)
# adding extra dimension to put this image into a batch by saying where we want to add this batch (as the first dimension)
test_image2 = np.expand_dims(test_image2, axis = 0)
# cnn prediction on the test image
result2 = cnn.predict(test_image2)

#prediction for the single image/element from the batch
if result2[0][0] == 1:
   prediction2 = 'dog'
else:
   prediction2 = 'cat'

print(prediction2)