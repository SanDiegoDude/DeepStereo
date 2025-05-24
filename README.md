# DeepStereo: AI-Powered Autostereogram Generator

DeepStereo is a Python script that generates autostereograms (like "Magic Eye" images) by first estimating depth from a standard 2D input image using an AI model (Intel's MiDaS), and then constructing the stereogram using a provided texture.

This allows you to turn almost any photograph into a 3D hidden image stereogram without needing to manually create a depth map.

## Features

*   **AI Depth Estimation:** Uses state-of-the-art MiDaS models (DPT_Large, DPT_Hybrid, MiDaS_small) from PyTorch Hub to automatically generate a depth map from your input image.
*   **Custom Textures:** Use any image as a repeating texture for the stereogram.
*   **Configurable Parameters:** Adjust minimum and maximum separation for the stereogram effect.
*   **Save Intermediate Depth Map:** Option to save the AI-generated depth map for inspection or reuse.

## Requirements

*   Python 3.7+
*   PyTorch
*   Torchvision
*   OpenCV-Python
*   Pillow (PIL Fork)
*   timm (PyTorch Image Models)

You can install these using the `requirements.txt` file:
`pip install -r requirements.txt`

## Usage

```bash
python deepstereo.py --input <path_to_your_color_image> \
                     --texture <path_to_your_texture_image> \
                     --output <path_for_the_generated_stereogram> \
                     [--midasmodel <model_name>] \
                     [--minsep <pixels>] \
                     [--maxsep <pixels>] \
                     [--save_depthmap <path_to_save_depthmap.png>]
