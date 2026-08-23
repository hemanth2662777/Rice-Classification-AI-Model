import tensorflow as tf

MODEL_PATH = "model/rice_model.h5"
TFLITE_PATH = "model/rice_model.tflite"

print("Loading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print("TFLite model created successfully.")
print(f"Saved to: {TFLITE_PATH}")