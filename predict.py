import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model("skin_disease_model.keras")

# Same class order used during training
class_names = [
    "Acne_Mild",
    "Acne_Moderate",
    "Acne_Severe",
    "Eczema_Mild",
    "Eczema_Moderate",
    "Eczema_Severe",
    "Normal"
]

# Test image path
image_path = r"D:\SD\Dataset\Dataset\test\Acne_Mild\182.JPG"

# Load image
img = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)

# Convert image to array
img_array = tf.keras.utils.img_to_array(img)

# Add batch dimension
img_array = tf.expand_dims(img_array, 0)

# Prediction
predictions = model.predict(img_array, verbose=0)

predicted_index = np.argmax(predictions[0])
confidence = np.max(predictions[0]) * 100

print("\nPredicted Disease:", class_names[predicted_index])
print("Confidence:", round(confidence, 2), "%")