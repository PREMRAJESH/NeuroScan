"""
Check the actual class order from ImageDataGenerator
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

BASE_DIR = Path(__file__).parent
TRAIN_DIR = BASE_DIR / 'brain_tumor_dataset' / 'Training'

print("Checking actual class order from flow_from_directory...")
print(f"Training directory: {TRAIN_DIR}\n")

# Create ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)

# Create flow_from_directory
train_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

print("Class indices from flow_from_directory:")
print(f"  {train_gen.class_indices}\n")

print("Class names (keys):")
class_names = list(train_gen.class_indices.keys())
for i, name in enumerate(class_names):
    print(f"  [{i}] {name}")
    
print("\nClass indices (mapping folder name to output index):")
for name, idx in sorted(train_gen.class_indices.items()):
    print(f"  {name} -> {idx}")
