# DeepStereo: AI-Powered Autostereogram Generator

DeepStereo is a Python script that generates autostereograms (like "Magic Eye" images) by first estimating depth from a standard 2D input image using an AI model (Intel's MiDaS), and then constructing the stereogram using a provided texture.

This allows you to turn almost any photograph into a 3D hidden image stereogram without needing to manually create a depth map.

## Features

*   **AI Depth Estimation:** Uses state-of-the-art MiDaS models (DPT_Large, DPT_Hybrid, MiDaS_small) from PyTorch Hub to automatically generate a depth map from your input image.
*   **Custom Textures:** Use any image as a repeating texture for the stereogram.
*   **Configurable Parameters:** Adjust minimum and maximum separation for the stereogram effect.
*   **Save Intermediate Depth Map:** Option to save the AI-generated depth map for inspection or reuse.
*   **GPU Acceleration:** Supports CUDA-enabled GPUs for significantly faster depth estimation.

## Requirements

*   Python 3.7+
*   PyTorch
*   Torchvision
*   Torchaudio
*   OpenCV-Python
*   Pillow (PIL Fork)
*   timm (PyTorch Image Models)

## Setup Instructions

It's highly recommended to use a Python virtual environment (e.g., `venv` or `conda`) for this project.

**1. Create and activate a virtual environment (optional but recommended):**

   *Using `venv` (standard Python):*
    ```bash
    python3 -m venv deepstereo_env
    source deepstereo_env/bin/activate  # On Linux/macOS
    # deepstereo_env\Scripts\activate  # On Windows
    ```

   *Using `conda`:*
    ```bash
    conda create -n deepstereo_env python=3.9 # Or your preferred Python version
    conda activate deepstereo_env
    ```

**2. Install PyTorch:**

   The version of PyTorch you need depends on whether you want to use a GPU (recommended for speed) or run CPU-only.

   *   **For GPU Support (NVIDIA CUDA):**
        This provides the best performance. You'll need an NVIDIA GPU with compatible drivers.
        It's best to get the precise command for your system (OS, package manager, CUDA version) directly from the official PyTorch website:
        [**PyTorch Get Started Page**](https://pytorch.org/get-started/locally/)

        For example, a common command for Linux with pip and CUDA 12.1 (check the website for the latest):
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```
        Ensure your NVIDIA drivers are up to date. For an RTX 4090, drivers from the 525.xx series or newer are recommended for CUDA 12.1 compatibility.

   *   **For CPU-Only:**
        If you don't have an NVIDIA GPU or prefer a CPU-only installation:
        Again, check the [PyTorch Get Started Page](https://pytorch.org/get-started/locally/) and select "CPU" as your compute platform.
        A typical command for Linux/macOS with pip:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ```
        For Windows with pip:
        ```bash
        pip install torch torchvision torchaudio
        ```
        (Windows CPU-only version is often the default if no CUDA version is specified in the pip install command directly from PyPI, but using the index URL from PyTorch website is safer).

**3. Install other dependencies:**

   Once PyTorch is installed, install the remaining packages using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
