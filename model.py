import pathlib
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


data_dir = pathlib.Path('images_original')

train_ds = tf.keras.utils.image_dataset_from_directory(data_dir,
                                                       validation_split=0.2,
                                                      subset='training',
                                                      image_size=(432,288),
                                                       seed=123,
                                                      batch_size=32)
val_ds = tf.keras.utils.image_dataset_from_directory(data_dir,
                                                    validation_split=0.2,
                                                    subset='validation',
                                                    seed=123,
                                                    image_size=(432,288),
                                                    batch_size=32)
class_names = train_ds.class_names
num_classes = len(class_names)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
model = Sequential([
    layers.Rescaling(1./255, input_shape=(432,288,3)),
    layers.Conv2D(16,3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])
model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
epochs = 5
model.fit(train_ds, validation_data=val_ds, epochs=epochs)
json_model = model.to_json()

with open('model.json', 'w') as json_file:
    json_file.write(json_model)

model.save_weights('model_weights.h5')
