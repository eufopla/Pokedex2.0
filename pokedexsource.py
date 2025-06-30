import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# LOAD THE LOCAL DATASET
data_dir = pathlib.Path(r"C:\Users\user\Desktop\Folder\Pokedex\dataset")
print(f"Dataset path: {data_dir.resolve()}")

batch_size = 8
img_height = 200
img_width = 200

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
val_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
class_names = train_data.class_names  
print("Classes détectées :", class_names)

# DISPLAY IMAGES
plt.figure(figsize=(10, 10))
for images, labels in train_data.take(1):
    for i in range(3):
        ax = plt.subplot(1, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

# MODEL CONSTRUCTION
num_classes = len(class_names)
model = tf.keras.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(128, 4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# EARLY STOP
early_stopping = EarlyStopping(
    monitor='val_loss',        # PERFORMANCE IS MONITORED ON VALIDATION
    patience=4,                # TOLERATE 4 EPOCHS WITHOUT IMPROVEMENT
    restore_best_weights=True 
)

# TRAINING
logdir = "logs"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[tensorboard_callback, early_stopping]
)

# PREDICTING
from tensorflow.keras.utils import load_img, img_to_array
image_path = r"C:\Users\user\Desktop\Projets\Pokedex\img_a_tester\test.jpg" #IMAGE TO PREDICT
if not os.path.exists(image_path):
    print(f"Image non trouvée : {image_path}")
    sys.exit(1)
image = load_img(image_path, target_size=(img_height, img_width))  # (200, 200)
image_array = img_to_array(image) / 255.0  # normalisation
image_array = np.expand_dims(image_array, axis=0)  # Shape: (1, 200, 200, 3)

plt.imshow(image)
plt.axis('off')
plt.title("Image à prédire")
plt.show()

predictions = model.predict(image_array)
res = np.argmax(predictions, axis=1)[0]

print(f"Classe prédite: {class_names[res]} ({res})")
print("Probabilités :", predictions)

# FILTER VISUALIZATION
def visualiser_filtres(name_image, model, layer_name, image):
    inp = model.inputs
    out1 = model.get_layer(layer_name).output
    feature_map_1 = Model(inputs=inp, outputs=out1)

    # RESIZE AND NORMALISE THE IMAGE
    img_resized = cv2.resize(image, (img_width, img_height))
    img_normalized = img_resized.astype("float32") / 255.0
    input_img = np.expand_dims(img_normalized, axis=0)

    f = feature_map_1.predict(input_img)
    dim = f.shape[3]
    print(f'{layer_name} | Features Shape: {f.shape}')
    print(f'Dimension: {dim}')
    
    fig = plt.figure(figsize=(30, 30))
    output_dir = f'results_{name_image}'
    os.makedirs(output_dir, exist_ok=True)
    for i in range(dim):
        ax = fig.add_subplot(dim//8, 8, i+1)
        ax.axis('off')
        ax.imshow(f[0, :, :, i], cmap='viridis')
        plt.imsave(f'{output_dir}/{name_image}_{layer_name}_{i}.jpg', f[0, :, :, i])

print(f"Classe prédite: {class_names[res]}")
