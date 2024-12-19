import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import defaultdict

# Custom Dataset class to load images
class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')  # convert to grayscale
        return self.transform(image)
    
def remove_png_extension(input_string):
    if input_string.endswith(".png"):
        return input_string[:-4]
    return input_string

def sum_images_in_batches(image_paths, batch_size=8):
    dataset = ImageDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # Use GPU if available
    device = torch.device('cuda')
    
    summed_images = None
    for batch in dataloader:
        batch = batch.to(device)
        if summed_images is None:
            summed_images = torch.sum(batch, dim=0)
        else:
            summed_images += torch.sum(batch, dim=0)
        
        del batch  # Free up GPU memory
        torch.cuda.empty_cache()

    return summed_images

def main(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Group images by session-scene-distortion
    image_groups = defaultdict(list)
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            user, session, scene, worldRotation, distortion, year, month, day, hour, minute, second = filename.split('-')
            distortion = remove_png_extension(distortion)
            key = f"{session}-{scene}-{distortion}"
            image_groups[key].append(os.path.join(input_dir, filename))

    # Process each group
    for key, image_paths in image_groups.items():
        summed_image = sum_images_in_batches(image_paths)
        summed_image = summed_image.cpu().numpy().astype(np.uint8)  # move to CPU and convert to uint8
        summed_image = Image.fromarray(summed_image.squeeze(), mode='L')  # convert to PIL image

        # Save summed image with appropriate filename
        output_path = os.path.join(output_dir, f"{key}.png")
        summed_image.save(output_path)

if __name__ == "__main__":
    input_dir = "fixmaps"
    output_dir = "sum_fixmaps"
    main(input_dir, output_dir)