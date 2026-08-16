import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==========================================
# 1. DATASET PATHS
# ==========================================

train_dir = r"D:\SD\Dataset\Dataset\train"
val_dir = r"D:\SD\Dataset\Dataset\val"
test_dir = r"D:\SD\Dataset\Dataset\test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# ==========================================
# 2. LOAD DATASETS
# ==========================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# 3. CLASS NAMES
# ==========================================

class_names = train_ds.class_names

print("\nClasses:")
print(class_names)

print("\nNumber of classes:", len(class_names))


# ==========================================
# 4. PERFORMANCE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ==========================================
# 5. DATA AUGMENTATION
# ==========================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1)
])


# ==========================================
# 6. MOBILE NET V2
# ==========================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False


# ==========================================
# 7. BUILD MODEL
# ==========================================

inputs = keras.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = keras.Model(
    inputs,
    outputs
)


# ==========================================
# 8. COMPILE
# ==========================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 9. MODEL SUMMARY
# ==========================================

model.summary()


# ==========================================
# 10. CALLBACKS
# ==========================================

callbacks = [

    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=0.000001
    )
]


# ==========================================
# 11. TRAINING
# ==========================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=callbacks
)


# ==========================================
# 12. TEST
# ==========================================

test_loss, test_accuracy = model.evaluate(
    test_ds
)

print("\n==============================")
print("FINAL TEST RESULT")
print("==============================")

print(
    "Test Accuracy:",
    test_accuracy
)

print(
    "Test Loss:",
    test_loss
)


# ==========================================
# 13. SAVE MODEL
# ==========================================

model.save(
    "skin_disease_model.keras"
)

print("\n==============================")
print("MODEL SAVED SUCCESSFULLY")
print("==============================")

print(
    "skin_disease_model.keras"
)