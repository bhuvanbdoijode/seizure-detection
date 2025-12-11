import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

IMG_SIZE = 128
BATCH_SIZE = 32

# Image generators
datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255
)

train_gen = datagen.flow_from_directory(
    "heatmap_images",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    subset="training",
    batch_size=BATCH_SIZE
)

val_gen = datagen.flow_from_directory(
    "heatmap_images",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    subset="validation",
    batch_size=BATCH_SIZE
)

# Build model
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(2, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15
)

model.save("seizure_heatmap_model.keras")
