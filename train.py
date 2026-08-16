import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =========================
# Dataset paths
# =========================
train_dir = r"D:\SD\Dataset\Dataset\train"
val_dir = r"D:\SD\Dataset\Dataset\val"
test_dir = r"D:\SD\Dataset\Dataset\test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# =========================
# Load datasets
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names

print("\nClasses:")
print(class_names)

# =========================
# Performance
# =========================
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# =========================
# Data Augmentation
# =========================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# =========================
# MobileNetV2 Base Model
# =========================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# =========================
# Build Model
# =========================
inputs = keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

# MobileNetV2 preprocessing
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

x = layers.Dense(128, activation="relu")(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = keras.Model(inputs, outputs)

# =========================
# Compile
# =========================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# Callbacks
# =========================
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    ),

    keras.callbacks.ModelCheckpoint(
        "best_skin_disease_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# =========================
# Training
# =========================
print("\nStarting Transfer Learning Training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

# =========================
# Test Evaluation
# =========================
test_loss, test_accuracy = model.evaluate(test_ds)

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")
print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

# =========================
# Save Final Model
# =========================
model.save("skin_disease_model_v2.keras")

print("\nModel saved as skin_disease_model_v2.keras")