import os
import numpy as np
import joblib  # Directly import joblib
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from skimage.io import imread
from skimage.filters import threshold_otsu

letters = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
    'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

def read_training_data(training_directory):
    image_data = []
    target_data = []
    for each_letter in letters:
        for each in range(10):
            image_path = os.path.join(training_directory, each_letter, f"{each_letter}_{each}.jpg")
            if not os.path.exists(image_path):
                continue  # Skip missing images
            
            img_details = imread(image_path, as_gray=True)
            binary_image = img_details < threshold_otsu(img_details)
            flat_bin_image = binary_image.reshape(-1)
            image_data.append(flat_bin_image)
            target_data.append(each_letter)

    return np.array(image_data), np.array(target_data)

def cross_validation(model, num_of_fold, train_data, train_label):
    accuracy_result = cross_val_score(model, train_data, train_label, cv=num_of_fold)
    print(f"Cross Validation Result for {num_of_fold}-fold: {accuracy_result * 100}")

current_dir = os.path.dirname(os.path.realpath(__file__))
training_dataset_dir = os.path.join(current_dir, 'train')

image_data, target_data = read_training_data(training_dataset_dir)

svc_model = SVC(kernel='linear', probability=True)
cross_validation(svc_model, 4, image_data, target_data)

svc_model.fit(image_data, target_data)

# Save the trained model
save_directory = os.path.join(current_dir, 'models', 'svc')
os.makedirs(save_directory, exist_ok=True)

joblib.dump(svc_model, os.path.join(save_directory, 'svc.pkl'))
print(f"Model saved to {os.path.join(save_directory, 'svc.pkl')}")
