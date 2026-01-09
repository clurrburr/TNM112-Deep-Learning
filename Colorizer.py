"""
Lab 3 – Image Colorization (TNM112)
Fully-convolutional U-Net trained to colorize grayscale images
Dataset: CIFAR-10
Framework: TensorFlow / Keras
"""


# =============================
# 1. Imports
# =============================
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np


# Säkerställ reproducerbarhet
tf.random.set_seed(42)
np.random.seed(42)


# =============================
# 2. Ladda CIFAR-10
# =============================
(x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()


# Normalisera till [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# =============================
# 3. Skapa grayscale-input
# =============================
def rgb_to_gray(images):
#Konverterar RGB-bilder till 1-kanals gråskala
    return tf.image.rgb_to_grayscale(images)


x_train_gray = rgb_to_gray(x_train)
x_test_gray = rgb_to_gray(x_test)


# =============================
# 4. Bygg U-Net-modell
# =============================
def build_unet():
    inputs = layers.Input(shape=(32, 32, 1))


# ----- Encoder -----
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)


    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)


# ----- Bottleneck -----
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(b)


# ----- Decoder -----
    u1 = layers.UpSampling2D()(b)
    u1 = layers.Concatenate()([u1, c2])
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(u1)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)


    u2 = layers.UpSampling2D()(c3)
    u2 = layers.Concatenate()([u2, c1])
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(c4)

# ----- Output -----
    outputs = layers.Conv2D(3, 1, activation='sigmoid')(c4)


    return models.Model(inputs, outputs)


model = build_unet()
model.summary()


# =============================
# 5. Kompilera modellen
# =============================
model.compile(
optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
loss='mse', # L2-loss
metrics=['mae'] # Mean Absolute Error
)


# =============================
# 6. Träna modellen
# =============================
history = model.fit(
x_train_gray,
x_train,
validation_split=0.1,
epochs=1,
batch_size=64,
verbose=1
)


# =============================
# 7. Utvärdering
# =============================
model.evaluate(x_test_gray, x_test)


# =============================
# 8. Visualisera resultat
# =============================
def plot_results(gray, pred, gt, n=5):
#Visar grayscale input, färgprediktion och ground truth
    plt.figure(figsize=(10, 6))
    for i in range(n):
# Grayscale
        plt.subplot(3, n, i + 1)
        plt.imshow(gray[i].squeeze(), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.ylabel('Input', fontsize=12)


# Prediction
        plt.subplot(3, n, i + 1 + n)
        plt.imshow(pred[i])
        plt.axis('off')
        if i == 0:
            plt.ylabel('Predicted', fontsize=12)


# Ground truth
        plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(gt[i])
        plt.axis('off')
        if i == 0:
            plt.ylabel('Ground truth', fontsize=12)


    plt.tight_layout()
    plt.show()


# Prediktera på testbilder
    preds = model.predict(x_test_gray[:5])
    plot_results(x_test_gray[:5], preds, x_test[:5])

def plot_before_after(gray, pred, gt, n=5):
    """
    Visualiserar:
    Rad 1: Grayscale input
    Rad 2: Modellens output
    Rad 3: Ground truth
    """
    plt.figure(figsize=(3*n, 8))

    for i in range(n):
        # Input (grayscale)
        plt.subplot(3, n, i + 1)
        plt.imshow(gray[i].squeeze(), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.ylabel("Input", fontsize=12)

        # Prediktion
        plt.subplot(3, n, i + 1 + n)
        plt.imshow(pred[i])
        plt.axis('off')
        if i == 0:
            plt.ylabel("Predicted", fontsize=12)

        # Ground truth
        plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(gt[i])
        plt.axis('off')
        if i == 0:
            plt.ylabel("Ground truth", fontsize=12)

    plt.tight_layout()
    plt.show()

    preds = model.predict(x_test_gray[:5])
    plot_before_after(x_test_gray[:5], preds, x_test[:5])

    # =============================
# 9. Spara modellen (valfritt)
# =============================
    model.save("colorization_unet_cifar10")