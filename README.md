---

# MASt3R-to-COLMAP: A Robust 3D Reconstruction Pipeline

This project provides a complete pipeline to leverage the state-of-the-art deep learning model **MASt3R** for robust initial 3D reconstruction and then seamlessly exports the result into a **COLMAP**-compatible format for final refinement and dense reconstruction.

It is designed to overcome common failure modes of traditional Structure-from-Motion (SfM) pipelines, such as scale drift and difficulties with textureless surfaces.

## The Problem: Limitations of Traditional SfM

Traditional SfM pipelines, while powerful, can struggle with:
*   **Sequential Processing:** They often add images one by one, which can lead to the accumulation of small errors, causing the reconstruction to "drift" or bend over long sequences.
*   **Scale Ambiguity:** Determining the true scale of a scene from images alone is impossible. This can lead to disjointed models or incorrect relative scales between different parts of a scene.
*   **Texture-less or Repetitive Surfaces:** Relying on handcrafted feature detectors (like SIFT) can fail in areas with uniform color or repetitive patterns.

## The Solution: Deep Learning for Global Alignment

This pipeline uses **MASt3R**, a model built on **DUST3R**, to address these issues.

1.  **Global Consistency:** Instead of processing images sequentially, MASt3R considers all image pairs at once to find a **globally optimal** alignment for all cameras simultaneously. This dramatically reduces drift and ensures a coherent final model.
2.  **Learned Features & Depth:** The model uses learned deep features and predicts a dense depth map for every pixel, allowing it to find correspondences even in low-texture regions where traditional methods fail.
3.  **Scaled Reconstruction:** By optimizing depth and poses jointly, MASt3R produces a sparse reconstruction that is not only globally consistent but also has a consistent (though arbitrary) scale.

This project's script acts as the critical bridge, taking the powerful initial model from MASt3R and converting it into the standard format used by COLMAP, the gold-standard tool for high-accuracy bundle adjustment and dense reconstruction.

## Prerequisites

1.  **Conda/Mamba:** For managing the Python environment.
2.  **Git and Git LFS:** For cloning the repository and handling large model files.
3.  **COLMAP:** You must have the COLMAP command-line interface installed and accessible in your system's PATH. ([Installation Guide](https://colmap.github.io/install.html))

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Set up the Conda Environment:**
    ```bash
    conda create -n mast3r python=3.10
    conda activate mast3r
    ```

3.  **Install Dependencies:**
    This project relies on the MASt3R environment. Install PyTorch and other dependencies as specified by the original repository.
    ```bash
    # Install PyTorch (adjust for your CUDA version if necessary)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    # Install other required packages
    pip install numpy opencv-python trimesh Pillow tqdm plyfile argparse
    ```

4.  **Clone MASt3R Repository:**
    The script relies on the MASt3R codebase. Clone it into your project directory.
    ```bash
    git clone https://github.com/naver/mast3r.git
    cd mast3r
    git lfs pull # Download the model weights
    cd ..
    ```

## Usage

### Step 1: Prepare Your Data

Place all your input images (e.g., `.jpg`, `.png`) in a single directory. For best results, use 5-10+ images with good overlap.

```
your_project/
|-- data/
|   |-- my_image_set/
|   |   |-- image001.jpg
|   |   |-- image002.jpg
|   |   |-- ...
|-- mast3r/
|-- your_script.py
`-- README.md
```

### Step 2: Run the MASt3R Reconstruction Script

Execute the main Python script to perform the initial reconstruction. This will create a `colmap` directory with the sparse model inside.

```bash
# Activate your environment
conda activate mast3r

# Run the script
python your_script.py \
    --image_dir data/my_image_set \
    --save_dir data/my_image_set_reconstruction \
    --model_path mast3r/mast3r_vitl14.pth \
    --device cuda \
    --shared_intrinsics
```
*   `--image_dir`: Path to your input images.
*   `--save_dir`: Path where the output `colmap` folder will be created.
*   `--shared_intrinsics`: Highly recommended flag for stability.

This will generate a folder structure like: `data/my_image_set_reconstruction/sparse/0/` containing `cameras.txt`, `images.txt`, and `points3D.txt`.

### Step 3: Refine with COLMAP Bundle Adjustment

This is a critical step for maximizing accuracy. Run COLMAP's bundle adjuster on the generated model.

```bash
colmap bundle_adjuster \
    --input_path data/my_image_set_reconstruction/sparse/0 \
    --output_path data/my_image_set_reconstruction/sparse/0
```
This command overwrites the sparse model with the refined, more accurate version.

### Step 4: (Optional) Create a Dense Point Cloud

Now that you have a high-quality sparse model, you can proceed with COLMAP's dense reconstruction pipeline.

1.  **Dense Stereo Matching:**
    ```bash
    colmap patch_match_stereo \
        --workspace_path data/my_image_set_reconstruction
    ```
    This creates a `stereo` sub-folder with depth and normal maps.

2.  **Stereo Fusion:**
    ```bash
    colmap stereo_fusion \
        --workspace_path data/my_image_set_reconstruction \
        --output_path data/my_image_set_reconstruction/dense.ply
    ```
    This fuses the depth maps into a final, dense point cloud named `dense.ply`.

## Troubleshooting

*   **Error: `Check failed: point3D.track.Length() > 1`**
    *   **Cause:** Your model contains 3D points that were observed in only one camera, which is invalid for bundle adjustment.
    *   **Solution:** The script should already filter these out. If you see this, ensure your script includes the logic to only save points where `len(track_obs) > 1`.

*   **Output: `Termination : No convergence`**
    *   **Cause:** The optimizer could not find a stable minimum, almost always because the scene geometry is weak (e.g., using only 2-3 images).
    *   **Solution:** This is not a fatal error, but a sign of a weak reconstruction. **Use more images (5-10+)** to provide stronger geometric constraints.

## To-Do / Future Work

- [ ] **Automate COLMAP Steps:** Integrate the `colmap bundle_adjuster` and dense reconstruction commands directly into the Python script using `subprocess` for a true one-click workflow.
- [ ] **Video Input:** Add functionality to accept a video file as input, using `ffmpeg` to extract frames automatically.
- [ ] **Configuration File:** Move command-line arguments to a `.yaml` or `.json` configuration file for easier management of complex runs.
- [ ] **Parameter Exposure:** Expose more of MASt3R's internal parameters (e.g., matching strategy) as command-line arguments.
- [ ] **Advanced Filtering:** Implement more advanced outlier filtering for the initial point cloud before passing it to COLMAP.

## Acknowledgments

This pipeline heavily relies on the incredible work from the authors of MASt3R and DUST3R. Please consider citing their original papers if you use this work in your research.

*   **MASt3R:** [Paper](https://arxiv.org/abs/2312.13324) | [GitHub](https://github.com/naver/mast3r)
*   **DUST3R:** [Paper](https://arxiv.org/abs/2312.14492) | [GitHub](https://github.com/naver/dust3r)
