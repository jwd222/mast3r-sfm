import rasterio
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm

# Set paths
input_dir = Path("/mnt/d/projects/wsl_projects/Projects/3_repo/mast3r/data/tobias/1/images")
output_dir = input_dir / "cropped"
output_dir.mkdir(exist_ok=True)

# Get all .tif files
tif_files = sorted(input_dir.glob("*.tif"))

# Step 1: Find minimum width and height
min_width = float('inf')
min_height = float('inf')

for tif_path in tif_files:
    with rasterio.open(tif_path) as src:
        min_width = min(min_width, src.width)
        min_height = min(min_height, src.height)

min_width = int(min_width)
min_height = int(min_height)
print(f"Cropping all rasters to size: {min_width} x {min_height}")

# Step 2: Crop and save
for tif_path in tqdm(tif_files, desc="Cropping"):
    with rasterio.open(tif_path) as src:
        window = Window(0, 0, min_width, min_height)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update({
            "height": min_height,
            "width": min_width,
            "transform": transform,
            "compress": None  # Ensure no compression
        })

        output_path = output_dir / tif_path.name
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(src.read(window=window))

print("✅ All images cropped and saved in:", output_dir)
