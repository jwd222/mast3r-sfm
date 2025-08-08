import pycolmap

recon = pycolmap.Reconstruction("/mnt/d/projects/wsl_projects/Projects/3_repo/mast3r/data/tobias/5/sparse/0")

# List all registered images
for image_id, image in recon.images.items():
    print(f"Image ID: {image_id}, Pose:\n{image.T}")

pass