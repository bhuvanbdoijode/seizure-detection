# train_cnn_attention.py
import os
import random
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------
# Reproducibility
# ---------------------------
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# ---------------------------
# Optional: mixed precision for speed (uncomment if desired & supported)
# ---------------------------
# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')

# ---------------------------
# Focal loss (sparse)
# ---------------------------
def sparse_categorical_focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = alpha * tf.pow((1 - y_pred), gamma)
        focal_loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
    return loss

# ---------------------------
# Lightweight augmentations for 1D signals/features (applied only to training)
# ---------------------------
def augment_1d(x, y, noise_std=0.02, shift_max=8):
    x = tf.cast(x, tf.float32)
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=noise_std, dtype=tf.float32)
    x = x + noise
    shift = tf.random.uniform([], minval=-shift_max, maxval=shift_max, dtype=tf.int32)
    x = tf.roll(x, shift=shift, axis=0)
    return x, y

# ---------------------------
# 1. LOAD & CLEAN DATA
# ---------------------------
print("Loading data...")
df = pd.read_csv("data.csv")
print("Loaded shape:", df.shape)

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# convert all feature columns to numeric
for col in df.columns:
    if col != "y":
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna().reset_index(drop=True)
print("After cleaning:", df.shape)

X = df.drop(columns=["y"]).values
y = df["y"].astype(int).values
y = y - 1  # labels 0..4

# ---------------------------
# 2. SPLIT: train / val / test (stratified)
# ---------------------------
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, random_state=SEED, stratify=y_trainval
)

print("Train / Val / Test shapes:", X_train.shape, X_val.shape, X_test.shape)

# ---------------------------
# 3. SCALE
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.joblib")
print("Saved scaler to scaler.joblib")

# reshape for Conv1D: (samples, length, 1)
X_train_cnn = X_train_scaled.reshape((-1, X_train_scaled.shape[1], 1))
X_val_cnn   = X_val_scaled.reshape((-1, X_val_scaled.shape[1], 1))
X_test_cnn  = X_test_scaled.reshape((-1, X_test_scaled.shape[1], 1))

# ---------------------------
# 4. CLASS WEIGHTS
# ---------------------------
classes = np.unique(y_train)
cw = class_weight.compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weights = {int(k): float(v) for k, v in zip(classes, cw)}
print("Class weights:", class_weights)

# ---------------------------
# 5. tf.data pipelines
# ---------------------------
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((X_train_cnn, y_train))
train_ds = train_ds.shuffle(2048, seed=SEED)
train_ds = train_ds.map(lambda x, y: augment_1d(x, y, noise_std=0.02, shift_max=8),
                        num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val_cnn, y_val)).batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test_cnn, y_test)).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# ---------------------------
# 6. MODEL: CNN encoder + MultiHeadAttention block
# ---------------------------
from tensorflow.keras import layers, Model

def build_cnn_attention(input_shape, n_classes=5, mha_heads=4, mha_key_dim=64, dropout_rate=0.3):
    inp = layers.Input(shape=input_shape)  # (length, 1)

    # --- CNN encoder (compact) ---
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(192, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    # no pooling here to preserve some sequence length
    x = layers.Dropout(0.2)(x)

    # --- Projection so attention has adequate channels ---
    proj = layers.Conv1D(mha_key_dim * mha_heads, kernel_size=1, padding="same", activation="relu")(x)
    proj = layers.LayerNormalization()(proj)  # normalize before attention

    # Multi-head attention expects (batch, seq_len, channels)
    # Use same tensor as query/key/value
    attn_output = layers.MultiHeadAttention(num_heads=mha_heads, key_dim=mha_key_dim)(
        proj, proj, proj
    )
    # Residual + LayerNorm
    attn_output = layers.Dropout(dropout_rate)(attn_output)
    attn_output = layers.Add()([proj, attn_output])
    attn_output = layers.LayerNormalization()(attn_output)

    # Optional small feed-forward block (position-wise)
    ff = layers.Dense(256, activation="relu")(attn_output)
    ff = layers.Dropout(dropout_rate)(ff)
    ff = layers.Dense(mha_key_dim * mha_heads, activation="relu")(ff)

    ff = layers.Add()([attn_output, ff])
    ff = layers.LayerNormalization()(ff)

    # Global pooling + classifier head
    pooled = layers.GlobalAveragePooling1D()(ff)
    pooled = layers.Dense(128, activation="relu")(pooled)
    pooled = layers.BatchNormalization()(pooled)
    pooled = layers.Dropout(0.4)(pooled)

    out = layers.Dense(n_classes, activation="softmax", dtype="float32")(pooled)

    model = Model(inputs=inp, outputs=out)
    return model

input_shape = (X_train_cnn.shape[1], 1)
model = build_cnn_attention(input_shape, n_classes=5, mha_heads=4, mha_key_dim=32, dropout_rate=0.25)
model.summary()

# ---------------------------
# 7. COMPILE
# ---------------------------
initial_lr = 3e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

model.compile(
    optimizer=optimizer,
    loss=sparse_categorical_focal_loss(gamma=2.0, alpha=0.25),
    metrics=["accuracy"]
)

# ---------------------------
# 8. CALLBACKS
# ---------------------------
logdir = "logs_attention"
os.makedirs(logdir, exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint("best_cnn_attention.keras", monitor="val_loss", save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.TensorBoard(log_dir=logdir)
]

# ---------------------------
# 9. TRAIN
# ---------------------------
EPOCHS = 60
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=2
)

# Save final model
model.save("final_cnn_attention.keras")
print("Saved final model to final_cnn_attention.keras")

# ---------------------------
# 10. PLOTS
# ---------------------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss Curve")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend()
plt.title("Accuracy Curve")
plt.grid(True)

plt.tight_layout()
plt.savefig("training_curves_attention.png", dpi=150)
plt.show()
print("Saved training_curves_attention.png")

# ---------------------------
# 11. EVALUATION
# ---------------------------
y_pred_prob = model.predict(X_test_cnn, batch_size=BATCH_SIZE)
y_pred = np.argmax(y_pred_prob, axis=1)

test_acc = accuracy_score(y_test, y_pred)
print("\nTest Accuracy:", test_acc)

print("\nClassification Report (Labels 1–5):")
print(classification_report(y_test + 1, y_pred + 1, digits=4))

cm = confusion_matrix(y_test + 1, y_pred + 1)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (5-Class)")
plt.tight_layout()
plt.savefig("confusion_matrix_attention.png", dpi=150)
plt.show()
print("Saved confusion_matrix_attention.png")

# Save predictions for analysis
pd.DataFrame({
    "y_true": (y_test + 1),
    "y_pred": (y_pred + 1),
}).to_csv("predictions_attention.csv", index=False)
print("Saved predictions_attention.csv")

# ---------------------------
# 12. Quick guidance
# ---------------------------
print("\nHints:")
print("- If validation loss still spikes: lower LR to 1e-4 or increase ReduceLROnPlateau patience.")
print("- If classes 1 & 2 still mix: try increasing mha_heads to 8 or mha_key_dim to 64, or increase augmentation diversity.")
print("- If overfitting appears: increase dropout in attention/feed-forward blocks to 0.35-0.5.")
