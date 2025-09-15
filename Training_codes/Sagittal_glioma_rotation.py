import os
from PIL import Image
from tqdm import tqdm

# Rotation angles
angles = [0, 45, 90, 135, 180, 225, 270, 315]

# Input and output folders
input_root = r"C:\Users\Sapna\Desktop\Sagittal_seperation\MRI-1\glioma"
output_root = r"C:\Users\Sapna\Desktop\Sagittal_seperation\MRI-1\glioma_rotated"

def rotate_and_save_images(input_root, output_root):
    for root, _, files in os.walk(input_root):
        # Maintain relative path to recreate folder structure
        rel_path = os.path.relpath(root, input_root)
        save_dir = os.path.join(output_root, rel_path)
        os.makedirs(save_dir, exist_ok=True)

        for file in tqdm(files, desc=f"Processing {rel_path}"):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(root, file)
                img = Image.open(img_path).convert("RGB")

                base, ext = os.path.splitext(file)

                # Rotate and save to new folder
                for angle in angles:
                    rotated = img.rotate(angle, expand=True, fillcolor=(0, 0, 0))  # black background
                    new_filename = f"{base}_rot{angle}{ext}"
                    rotated.save(os.path.join(save_dir, new_filename))

if __name__ == "__main__":
    rotate_and_save_images(input_root, output_root)
