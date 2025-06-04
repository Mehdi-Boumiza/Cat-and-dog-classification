training_dir = "/Users/mehdiboumiza/Documents/learning00/archive/training_set/training_set"
testing_dir = "/Users/mehdiboumiza/Documents/learning00/archive/test_set/test_set"
import tensorflow as tf
train_data = tf.keras.utils.image_dataset_from_directory(
    training_dir,
    image_size = (128,128),
    batch_size = 128,
    label_mode = 'binary',
    subset = 'training',
    seed = 123,
    validation_split = 0.2
)

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation

data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
])
train_data = train_data.map(lambda x, y: (data_augmentation(x), y))


val_data = tf.keras.utils.image_dataset_from_directory(
    training_dir,
    image_size = (128,128),
    batch_size = 128,
    label_mode='binary',
    subset = 'validation',
    seed = 123,
    validation_split = 0.2
)
test_data = tf.keras.utils.image_dataset_from_directory(
    testing_dir,
    image_size = (128,128),
    batch_size = 128,
    label_mode = 'binary',
)

normalization_layer = Rescaling(1./255)

train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))
test_data = test_data.map(lambda x, y: (normalization_layer(x), y))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout
model = Sequential([
    Conv2D(32,(3,3), activation = 'relu', input_shape = [128,128,3]),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(256,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(256,activation='relu'),
    Dense(1,activation='sigmoid'),
])
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)
history = model.fit(train_data,validation_data = val_data,epochs = 40)
model.evaluate(test_data)


model.save("cat_dog_classifier.h5")
import matplotlib.pyplot as plt

plt.plot(history.history['Accuracy'], label='Train Accuracy')
plt.plot(history.history['Val_Accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()