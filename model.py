import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('/Users/mehdiboumiza/Documents/learning00/cat_dog_classifier.h5')

from tensorflow.keras.preprocessing import image
import numpy as np
img = image.load_img("/Users/mehdiboumiza/Documents/learning00/images (4).jpeg", target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0  
img_array = np.expand_dims(img_array, axis=0)  

prediction = model.predict(img_array)
if prediction[0][0] > 0.5:
    print("It's a dog ")
else:
    print("It's a cat ")
