from tensorflow.keras.models import load_model

try:
    model = load_model("model/rice_model.h5", compile=False)

    print("SUCCESS")
    print(model.summary())

except Exception as e:
    print("FAILED")
    print(e)