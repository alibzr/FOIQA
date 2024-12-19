import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image

def remove_png_extension(input_string):
    if input_string.endswith(".png"):
        return input_string[:-4]
    return input_string

# Configuration
first_dir = 'binary_masks'  # should be 60 images
second_dir = 'sum_fixmaps'  # should be 70 images
output_dir = 'masked_fixmaps'  # should result in 4200 images
batch_size = 1  # Number of images to process at once (adjust based on your GPU memory)

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Transformation to convert images to tensors
transform = transforms.ToTensor()

# Load images from the first directory
first_images_list = sorted([os.path.join(first_dir, f) for f in os.listdir(first_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
second_images_list = sorted([os.path.join(second_dir, f) for f in os.listdir(second_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

# Move datasets to GPU
device = torch.device('cuda')

# Function to load and process images in batches
def load_images(image_list):
    images = []
    names = []
    for path in image_list:
        image = Image.open(path).convert("RGB")  # Ensure the image is in RGB mode
        image = transform(image).to(device)  # Apply transforms and move to GPU
        images.append(image)
        names.append(os.path.splitext(os.path.basename(path))[0])
    return images, names

# Function to multiply and save images
def multiply_and_save_images(first_images, second_images, first_names, second_names):
    for i, first_image in enumerate(first_images):
        for j, second_image in enumerate(second_images):
            # Multiply images
            multiplied_image = first_image * second_image

            # Clip values to valid range [0, 1]
            multiplied_image = torch.clamp(multiplied_image, 0, 1)

            # Convert back to CPU and then to PIL Image
            multiplied_image = multiplied_image.mul(255).byte().cpu()
            multiplied_image = transforms.ToPILImage()(multiplied_image)

            # Create the output filename by concatenating names
            output_filename = f"{first_names[i]}-{second_names[j]}.png"
            output_path = os.path.join(output_dir, output_filename)

            # Save the image
            multiplied_image.save(output_path)

# Process images in batches
for i in range(0, len(first_images_list), batch_size):
    first_batch_paths = first_images_list[i:i + batch_size]
    first_images, first_names = load_images(first_batch_paths)

    for j in range(0, len(second_images_list), batch_size):
        second_batch_paths = second_images_list[j:j + batch_size]
        second_images, second_names = load_images(second_batch_paths)

        # Multiply and save images
        multiply_and_save_images(first_images, second_images, first_names, second_names)

print("Image multiplication and saving completed!")


