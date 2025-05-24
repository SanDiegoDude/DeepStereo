# DeepStereo: AI-Powered Autostereogram Generator

DeepStereo is a Python tool that generates autostereograms (like "Magic Eye" images) by first estimating depth from a standard 2D input image using AI models (Intel's MiDaS), and then constructing the stereogram using various algorithms and textures.

This allows you to turn almost any photograph into a 3D hidden image stereogram without needing to manually create a depth map.

![image](https://github.com/user-attachments/assets/a08ebb8f-0301-4e17-bb23-bfed131dd1df)



## Features

### Core Features
*   **AI Depth Estimation:** Uses state-of-the-art MiDaS models (DPT_Large, DPT_Hybrid, MiDaS_small) from PyTorch Hub to automatically generate a depth map from your input image.
*   **Multiple Stereogram Algorithms:**
    - Standard texture-based stereogram
    - Improved algorithm with better texture continuity (`--tex_alt_algo`)
    - Layered algorithm with texture preprocessing to reduce repetition (`--tex_alt_layer`)
    - Random Dot Stereogram (RDS) for classic Magic Eye style (`--tex_alt_rds`)
*   **GPU Acceleration:** Supports CUDA-enabled GPUs for significantly faster depth estimation.

### Texture Generation Features
*   **On-the-fly Texture Generation:** Automatically generate textures from your input image using multiple methods
*   **Texture Generation Methods:**
    - **Method 1:** Content-driven color dots with various color modes
    - **Method 2:** Density/size-driven patterns based on image brightness
    - **Method 3:** Voronoi/Worley noise patterns
    - **Method 4:** Stylized dithering with glyphs
*   **Texture Transforms:** Rotate, grid, and invert colors of any texture
*   **Custom Textures:** Use any image as a repeating texture for the stereogram
*   **Standalone Texture Generator:** Use `deeptexture.py` separately to create textures

### Output Options
*   **Configurable Parameters:** Adjust minimum and maximum separation for the stereogram effect
*   **Save Intermediate Results:** Option to save the AI-generated depth map and generated textures
*   **Flexible Output Naming:** Automatic descriptive filenames based on your settings

## Requirements

*   Python 3.7+
*   PyTorch
*   Torchvision
*   Torchaudio
*   OpenCV-Python
*   Pillow (PIL Fork)
*   NumPy
*   tqdm
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
        Get the precise command for your system from: [**PyTorch Get Started Page**](https://pytorch.org/get-started/locally/)

        For example, for Linux with pip and CUDA 12.1:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```

   *   **For CPU-Only:**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ```

**3. Install other dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Basic Usage

### Simple Example
Generate a stereogram using default settings:
```bash
python deepstereo.py --input path/to/your/image.jpg
```

### Using Different Stereogram Algorithms

**Standard Algorithm (default):**
```bash
python deepstereo.py --input image.jpg
```

**Improved Algorithm (better texture flow):**
```bash
python deepstereo.py --input image.jpg --tex_alt_algo
```

**Layered Algorithm (reduced repetition):**
```bash
python deepstereo.py --input image.jpg --tex_alt_layer
```

**Random Dot Stereogram:**
```bash
python deepstereo.py --input image.jpg --tex_alt_rds --dot_density 0.6
```

### Using Custom Textures

**Load external texture:**
```bash
python deepstereo.py --input image.jpg --texture path/to/texture.jpg
```

**Transform the texture:**
```bash
python deepstereo.py --input image.jpg --texture texture.jpg --tex_rotate 45 --tex_grid 3,3 --tex_invert_colors
```

### On-the-fly Texture Generation

**Use default auto-generation (Method 1 + Method 4 blend):**
```bash
python deepstereo.py --input image.jpg --generate_texture_on_the_fly
```

**Specify texture generation methods:**
```bash
python deepstereo.py --input image.jpg --tex_method1_color_dots --tex_m1_color_mode transformed_hue
```

**Combine multiple methods:**
```bash
python deepstereo.py --input image.jpg --tex_method1_color_dots --tex_method4_glyph_dither --tex_combination_mode blend --tex_blend_type overlay
```

### Advanced Options

**Save depth map and generated texture:**
```bash
python deepstereo.py --input image.jpg --save_depthmap output_dir --save_generated_texture
```

**Use larger MiDaS model for better depth:**
```bash
python deepstereo.py --input image.jpg --midasmodel DPT_Large
```

**Adjust stereogram parameters:**
```bash
python deepstereo.py --input image.jpg --minsep 30 --maxsep 120
```

## Standalone Texture Generator

You can use `deeptexture.py` independently to create textures:

```bash
python deeptexture.py --input source_image.jpg --tex_method1_color_dots --tex_m1_dot_size 5
```

## Command Line Options

### Main Options
- `--input`: Path to input image (required)
- `--output_dir`: Output directory (default: "output_stereograms")
- `--output_filename_base`: Custom base name for output files

### Stereogram Generation
- `--minsep`: Minimum separation for far points (default: 40)
- `--maxsep`: Maximum separation for near points (default: 100)
- `--tex_alt_algo`: Use improved texture algorithm
- `--tex_alt_layer`: Use layered algorithm with preprocessing
- `--tex_alt_rds`: Generate random dot stereogram
- `--dot_density`: Dot density for RDS (0.0-1.0, default: 0.5)

### Depth Map Generation
- `--midasmodel`: Choose MiDaS model (MiDaS_small, DPT_Large, DPT_Hybrid)
- `--save_depthmap`: Save the generated depth map
- `--depth_model_input_width`: Resize width for MiDaS processing
- `--depth_invert`: Invert the depth map

### Texture Options
- `--texture`: Path to external texture image
- `--generate_texture_on_the_fly`: Force texture generation from input
- `--tex_input_raw`: Use raw input image as texture
- `--tex_rotate`: Rotate texture (degrees)
- `--tex_grid`: Create grid from texture (rows,cols)
- `--tex_invert_colors`: Invert texture colors

### Texture Generation Methods
See `python deepstereo.py --help` for full list of texture generation options including methods 1-4 and their parameters.

## Tips for Best Results

1. **Input Images:** High-contrast images with clear foreground/background separation work best
2. **Texture Choice:** For standard algorithms, use textures that tile well. For RDS, no texture needed
3. **Viewing:** Cross your eyes or look "through" the image to see the 3D effect
4. **Depth Models:** DPT_Large provides the best depth maps but is slower than MiDaS_small
5. **Separation Values:** Adjust `--minsep` and `--maxsep` based on your viewing distance and image size

## Examples Gallery

Check the `examples/` directory for sample outputs and the commands used to generate them.

## Troubleshooting

- **CUDA Out of Memory:** Use `--midasmodel MiDaS_small` or reduce `--depth_model_input_width`
- **Can't see 3D effect:** Try adjusting `--minsep` and `--maxsep` values
- **Repetitive patterns:** Use `--tex_alt_layer` or `--tex_alt_rds` for better results

## License

This project is open source. See LICENSE file for details.

## Acknowledgments

- Intel ISL for the MiDaS depth estimation models
- The computer vision community for stereogram algorithms
