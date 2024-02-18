import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import shutil
# Original dataset path
original_dataset = r"D:\HerbEsentia\HerbEsentia Modal\dataset"

# Directories for train, validation, and test sets
train_dataset = r"D:\HerbEsentia\HerbEsentia Modal\train"
validation_dataset = r"D:\HerbEsentia\HerbEsentia Modal\validation"
test_dataset = r"D:\HerbEsentia\HerbEsentia Modal\test"

# Create train, validation, and test directories
for dataset in [train_dataset, validation_dataset, test_dataset]:
    os.makedirs(os.path.join(dataset, 'plants'), exist_ok=True)
    os.makedirs(os.path.join(dataset, 'leaves'), exist_ok=True)

# Split dataset into train, validation, and test sets
data_path = Path(original_dataset)

# Assuming binary classification (plants or leaves).
images = []
labels = []

for category in ['plants', 'leaves']:
    category_path = data_path / category
    subcategories = os.listdir(category_path)

    for subcategory in subcategories:
        subcategory_path = category_path / subcategory
        image_paths = [str(subcategory_path / img) for img in os.listdir(subcategory_path) if img.endswith('.jpg')]
        images.extend(image_paths)
        labels.extend([1 if category == 'plants' else 0] * len(image_paths))

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Move images to respective directories using shutil
def copy_images_to_directory(image_paths, labels, dataset_path):
    for image_path, label in zip(image_paths, labels):
        category = 'plants' if label == 1 else 'leaves'
        destination = os.path.join(dataset_path, category, os.path.basename(image_path))
        shutil.copy(image_path, destination)

# Copy images to train, validation, and test sets
copy_images_to_directory(X_train, y_train, train_dataset)
copy_images_to_directory(X_val, y_val, validation_dataset)
copy_images_to_directory(X_test, y_test, test_dataset)

# Model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Image preprocessing function
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    return img_array

# Train the model using ImageDataGenerator.flow_from_directory
train_data_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dataset,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
)

model.fit(train_data_generator, epochs=10)

# Evaluate the model
validation_data_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    validation_dataset,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
)
validation_accuracy = model.evaluate(validation_data_generator)
print('Validation Accuracy:', validation_accuracy[1])
# Save the model in the native Keras format
model.save('trained_models/plant_classification_model.h5')
