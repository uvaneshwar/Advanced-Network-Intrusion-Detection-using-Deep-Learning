import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

# Function to convert numerical data to image
def numerical_to_image(data, img_size=(28, 28)):
    # Reshape the data to fit the image size
    # Calculate the required padding
    padding_size = img_size[0] * img_size[1] - data.size
    # Add padding to the data if necessary
    padded_data = np.pad(data, (0, padding_size), 'constant')
    reshaped_data = np.reshape(padded_data, img_size)
    return reshaped_data

# Convert all the LSTM outputs to images
images = np.array([numerical_to_image(lstm_output[i], (28, 28)) for i in range(len(lstm_output))])

# Create a directory to save images
save_dir = 'lstm_images'
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Save each image
for i in range(len(images)):
    img = (images[i] * 255).astype(np.uint8)  # Scale to 0-255 for image saving
    img_pil = Image.fromarray(img)
    img_pil.save(os.path.join(save_dir, f'lstm_image_{i}.png'))

print(f'LSTM images saved to {save_dir}/')

# Example: visualize one image
plt.imshow(images[0], cmap='gray')
plt.title('Example LSTM Image')
plt.axis('off')  # Hide axis
plt.show()

# Optional: If using Google Colab, zip and download the images
import shutil
from google.colab import files

# Zip the directory
shutil.make_archive('lstm_images', 'zip', save_dir)

# Download the zip file
files.download('lstm_images.zip')
