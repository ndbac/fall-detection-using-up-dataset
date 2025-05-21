import cv2
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# === Load the model ===
model_path = 'model_combined.keras'
model = load_model(model_path)
print(f"✅ Model loaded from {model_path}!")

# === Define class labels ===
class_labels = [
    'Unknown',
    'Falling hands',
    'Falling knees',
    'Falling backwards',
    'Falling sideward',
    'Falling chair',
    'Normal',
    'Normal',
    'Normal',
    'Normal',
    'Normal',
    'Laying'
]

# === Test folder path ===
test_folder = 'test/actual'

if not os.path.exists(test_folder):
    raise FileNotFoundError(f"❌ Test folder '{test_folder}' not found!")

# === Get image files ===
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
image_files = [f for f in os.listdir(test_folder)
               if os.path.isfile(os.path.join(test_folder, f)) and
               any(f.lower().endswith(ext) for ext in image_extensions)]

if not image_files:
    raise RuntimeError(f"❌ No image files found in '{test_folder}'")

print(f"🖼️ Found {len(image_files)} test images.")

# === Preprocess all images and store them ===
processed_images = []
file_names = []

for img_file in image_files:
    img_path = os.path.join(test_folder, img_file)
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ Skipped corrupted file: {img_file}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32)).astype('float32') / 255.0
    input_image = resized.reshape(32, 32, 1)
    processed_images.append(input_image)
    file_names.append(img_file)

processed_images = np.array(processed_images)

# === Make predictions on all images (dummy dual input: cam1 = cam2) ===
predictions = model.predict(processed_images, verbose=0)
pred_classes = np.argmax(predictions, axis=1)

# === Display each result using OpenCV with overlay ===
for i, img_file in enumerate(file_names):
    img_path = os.path.join(test_folder, img_file)
    image = cv2.imread(img_path)
    class_index = pred_classes[i]
    class_name = class_labels[class_index]
    confidence = predictions[i][class_index] * 100

    status_color = (0, 255, 0)  # Green
    if 'Falling' in class_name:
        status_color = (0, 0, 255)  # Red for danger class

    cv2.putText(image, f"Class: {class_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(image, f"Confidence: {confidence:.2f}%", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    # Resize for display if large
    h, w = image.shape[:2]
    if h > 800 or w > 800:
        scale = min(800 / h, 800 / w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    cv2.imshow(f"Prediction - {img_file}", image)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
    cv2.destroyAllWindows()

print("✅ Test completed.")
