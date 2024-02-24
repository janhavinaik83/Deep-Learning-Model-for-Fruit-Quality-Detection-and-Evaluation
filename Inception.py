import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

# Define data paths and image size
data_dir = "path/to/your/fruit/images"  # Replace with your directory
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "validation")
img_size = (224, 224)  # Adjust based on InceptionV3 input size

# Create data generators with efficient CPU-friendly augmentations
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.05,  # Reduce shear range for CPU efficiency
    zoom_range=0.05,  # Reduce zoom range for CPU efficiency
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    data_format="channels_last"  # Ensure channels last for CPU compatibility
)
val_datagen = ImageDataGenerator(rescale=1./255, data_format="channels_last")

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=32, class_mode="categorical"
)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=32, class_mode="categorical"
)

# Load pre-trained InceptionV3, freeze earlier layers, and add new classifier
base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))
for layer in base_model.layers[:15]:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(512, activation="relu")(x)  # Consider smaller dense layers for CPU efficiency
predictions = Dense(len(your_fruit_classes), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile with CPU-friendly optimizer and loss
model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

# Train with early stopping and checkpointing for CPU efficiency
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(monitor="val_accuracy", patience=3)
checkpoint = ModelCheckpoint("best_model.h5", monitor="val_accuracy", save_best_only=True)

history = model.fit(
    train_generator, steps_per_epoch=len(train_generator), epochs=10,
    validation_data=val_generator, validation_steps=len(val_generator),
    callbacks=[early_stopping, checkpoint]
)

# Load the best model and evaluate on test set
best_model = tf.keras.models.load_model("best_model.h5")
test_datagen = ImageDataGenerator(rescale=1./255, data_format="channels_last")
test_generator = test_datagen.flow_from_directory(
    "path/to/test/images", target_size=img_size, batch_size=32, class_mode="categorical"
)
test_loss, test_acc = best_model.evaluate(test_generator)
print("Test accuracy:", test_acc)

# Make prediction on a new image
new_img = Image.open("path/to/new/image.jpg").resize(img_size)
new_img_array = np.array(new_img) / 255.0
new_img_array = np.expand_dims(new_img_array, axis=0)

prediction = best_model.predict(new_img_array)
print("Predicted class:", np.argmax(prediction[0]))
