import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

# Load and preprocess images using ImageDataGenerator
data_path = Path('dataset')
image_generator = ImageDataGenerator(rescale=1./255)

# Assuming binary classification (medicinal or non-medicinal)
images = []
labels = []

# Debugging: Print the list of files found
for category in ['plants', 'leaves']:
    category_path = data_path / category
    subcategories = os.listdir(category_path)
    
    for subcategory in subcategories:
        subcategory_path = category_path / subcategory
        image_paths = [str(subcategory_path / img) for img in os.listdir(subcategory_path) if img.endswith('.jpg')]
        print(f"Found {len(image_paths)} images in {category}/{subcategory} category.")
        images.extend(image_paths)
        labels.extend([1 if category == 'plants' else 0] * len(image_paths))

# Debugging: Print the first 5 image paths
print("Sample image paths:", images[:5])

# Check if the dataset is not empty
if not images:
    raise ValueError("The dataset is empty. Please check the data path and ensure there are images.")

# Adjust the test_size or train_size to avoid an empty train set
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Check if the train set is not empty
if not X_train:
    raise ValueError("The train set is empty. Adjust test_size or train_size parameters.")

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
train_data_generator = image_generator.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training',
)

model.fit(train_data_generator, epochs=10)

# Evaluate the model
test_data_generator = image_generator.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
)

test_accuracy = model.evaluate(test_data_generator)
print('Test Accuracy:', test_accuracy[1])

# Save the model
model.save('trained_models/medicinal_model.h5')
