import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# =========================
# Paths
# =========================
test_dir = r"D:\SD\Dataset\Dataset\test"
model_path = "skin_disease_model_v2.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# =========================
# Load test dataset
# =========================
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("\nClasses:")
print(class_names)

# =========================
# Load trained model
# =========================
model = tf.keras.models.load_model(model_path)

# =========================
# Predictions
# =========================
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# =========================
# Classification Report
# =========================
print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(y_true, y_pred)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(cm)

# =========================
# Plot Confusion Matrix
# =========================
plt.figure(figsize=(10, 8))

plt.imshow(cm)

plt.title("Skin Disease Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=90
)

plt.yticks(
    range(len(class_names)),
    class_names
)

plt.colorbar()

plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)

plt.show()

print("\nConfusion matrix saved as confusion_matrix.png")