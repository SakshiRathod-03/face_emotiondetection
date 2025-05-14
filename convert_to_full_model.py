# ✅ STEP 1: SABSE PEHLE Imports
from tensorflow.keras.models import model_from_json
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import Sequential

# ✅ STEP 2: Sequential ko register karna
@register_keras_serializable()
class MySequential(Sequential):
    pass

# ✅ STEP 3: JSON model load karna
with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()

# ✅ STEP 4: Model deserialize karna
model = model_from_json(model_json, custom_objects={"Sequential": MySequential})

# ✅ STEP 5: Weights load karna
model.load_weights("facialemotionmodel.h5")

# ✅ STEP 6: Save full model for future use
model.save("full_facial_model.h5")

print("✅ Model loaded and saved as full_facial_model.h5")
