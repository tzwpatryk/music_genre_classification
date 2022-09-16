import pathlib

import tensorflow as tf
import numpy as np

with open('model.json', 'r') as json_file:
    json_savedModel = json_file.read()
model_j = tf.keras.models.model_from_json(json_savedModel)

model_j.load_weights('model_weights.h5')

test_image = tf.keras.utils.load_img('img.png', target_size=(432, 288))
test_image = tf.keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model_j.predict(test_image)
print(np.array(tf.nn.softmax(result))[0])

data_dir = pathlib.Path('images_original')

train_ds = tf.keras.utils.image_dataset_from_directory(data_dir,
                                                       validation_split=0.2,
                                                      subset='training',
                                                      image_size=(432,288),
                                                       seed=123,
                                                      batch_size=32)
class_names = train_ds.class_names
print(class_names)