import os
import shutil

def copy_fixmap_files(source_path, destination_path):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # Iterate over all the folders in the source path
    for folder_name in os.listdir(source_path):
        folder_path = os.path.join(source_path, folder_name)
        
        # Check if the path is a directory
        if os.path.isdir(folder_path):
            fixmap_file = os.path.join(folder_path, "_fixmap.png")
            
            # Check if the _fixmap.png file exists in the folder
            if os.path.isfile(fixmap_file):
                # Copy the _fixmap.png file to the destination folder
                destination_file = os.path.join(destination_path, f"{folder_name}.png")
                shutil.copy(fixmap_file, destination_file)
                print(f"Copied: {fixmap_file} to {destination_file}")

# Define the source and destination paths
source_path = 'processed_gaze'
destination_path = 'fixmaps'

# Call the function to copy the files
copy_fixmap_files(source_path, destination_path)
