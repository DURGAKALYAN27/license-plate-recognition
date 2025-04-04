import os
import joblib  
import segmentation

# Load the trained model
current_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(current_dir, 'models/svc/svc.pkl')

# Check if the model exists before loading
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model file not found: {model_dir}")

model = joblib.load(model_dir)

classification_result = []

# Ensure characters exist before processing
if not hasattr(segmentation, "characters") or len(segmentation.characters) == 0:
    raise ValueError("No characters found for segmentation!")

for each_character in segmentation.characters:
    # Reshape and predict
    each_character = each_character.reshape(1, -1)
    result = model.predict(each_character)
    classification_result.append(result[0])  # Extract actual predicted value

# Convert list to string
plate_string = ''.join(classification_result)
print("Predicted Plate (Unsorted):", plate_string)

# Ensure column list is available
if not hasattr(segmentation, "column_list") or len(segmentation.column_list) == 0:
    raise ValueError("Column list for sorting characters is missing!")

# Correct character order using column positions
column_list_copy = segmentation.column_list[:]
segmentation.column_list.sort()

rightplate_string = ''.join(
    plate_string[column_list_copy.index(each)] for each in segmentation.column_list
)

print("Final License Plate:", rightplate_string)
