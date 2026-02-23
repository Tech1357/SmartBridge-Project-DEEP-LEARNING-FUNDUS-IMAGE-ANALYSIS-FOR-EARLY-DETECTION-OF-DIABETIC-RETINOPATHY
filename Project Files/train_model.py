"""
Deep Learning Fundus Image Analysis for Early Detection of Diabetic Retinopathy
Training Script - Xception Model
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

print("="*70)
print("DIABETIC RETINOPATHY DETECTION - MODEL TRAINING")
print("="*70)

# Configuration
TRAIN_PATH = 'data'
TEST_PATH = 'data'
IMG_SIZE = 299
BATCH_SIZE = 32  # Reduced for better gradient updates
EPOCHS = 50  # Increased epochs
NUM_CLASSES = 5

print(f"\nConfiguration:")
print(f"  Training Path: {TRAIN_PATH}")
print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Classes: {NUM_CLASSES}")
print(f"\nTensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Data Augmentation
print("\n" + "="*70)
print("STEP 1: Configuring Data Augmentation")
print("="*70)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

print("✅ Data augmentation configured")

# Load Data
print("\n" + "="*70)
print("STEP 2: Loading Training and Validation Data")
print("="*70)

train_generator = train_datagen.flow_from_directory(
    directory=TRAIN_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory=TEST_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

print(f"\n✅ Data loaded successfully")
print(f"  Class Indices: {train_generator.class_indices}")
print(f"  Training Samples: {train_generator.samples}")
print(f"  Validation Samples: {test_generator.samples}")
print(f"  Steps per Epoch: {train_generator.samples // BATCH_SIZE}")
print(f"  Validation Steps: {test_generator.samples // BATCH_SIZE}")

# Build Model
print("\n" + "="*70)
print("STEP 3: Building Xception Model")
print("="*70)

base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Unfreeze the last 20 layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D(name='global_avg_pooling')(x)
x = Dense(1024, activation='relu', name='dense_1024')(x)
x = Dropout(0.5, name='dropout_1')(x)
x = Dense(512, activation='relu', name='dense_512')(x)
x = Dropout(0.4, name='dropout_2')(x)
x = Dense(256, activation='relu', name='dense_256')(x)
x = Dropout(0.3, name='dropout_3')(x)
predictions = Dense(NUM_CLASSES, activation='softmax', name='output')(x)

model = Model(inputs=base_model.input, outputs=predictions)

print("✅ Model architecture created")
print(f"  Base Model: Xception (last 20 layers unfrozen for fine-tuning)")
print(f"  Custom Layers: GlobalAvgPool → Dense(1024) → Dense(512) → Dense(256) → Dense({NUM_CLASSES})")

# Compile Model
print("\n" + "="*70)
print("STEP 4: Compiling Model")
print("="*70)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model compiled")
print("  Optimizer: Adam (lr=0.0001)")
print("  Loss: categorical_crossentropy")
print("  Metrics: accuracy")

# Callbacks
print("\n" + "="*70)
print("STEP 5: Setting up Callbacks")
print("="*70)

checkpoint = ModelCheckpoint(
    'model/Updated-Xception-diabetic-retinopathy.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-7
)

callbacks = [checkpoint, early_stop, reduce_lr]

print("✅ Callbacks configured")
print("  - ModelCheckpoint: Save best model")
print("  - EarlyStopping: Stop if no improvement (patience=10)")
print("  - ReduceLROnPlateau: Reduce LR on plateau (patience=5)")

# Train Model
print("\n" + "="*70)
print("STEP 6: TRAINING MODEL")
print("="*70)
print("\n🚀 Starting training... This will take 2-4 hours with GPU\n")

steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = test_generator.samples // BATCH_SIZE

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

# Save Model
print("\n" + "="*70)
print("STEP 7: Saving Model")
print("="*70)

model_path = 'model/Updated-Xception-diabetic-retinopathy.h5'
model.save(model_path)
print(f"✅ Model saved to: {model_path}")

# Evaluate
print("\n" + "="*70)
print("STEP 8: Final Evaluation")
print("="*70)

test_loss, test_accuracy = model.evaluate(test_generator, steps=validation_steps)

print(f"\n{'='*70}")
print("TRAINING COMPLETE!")
print(f"{'='*70}")
print(f"\n📊 Final Results:")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"\n📈 Training History:")
print(f"  Best Validation Loss: {min(history.history['val_loss']):.4f}")
print(f"  Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"\n💾 Model saved at: {model_path}")
print(f"\n✅ Ready to run Flask app: python app.py")
print(f"{'='*70}\n")
