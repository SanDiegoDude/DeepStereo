import argparse
from PIL import Image, ImageOps
import torch
import numpy as np
import cv2 # OpenCV for transformations

# Constants for stereogram generation (can be args later)
MIN_SEPARATION_DEFAULT = 70
MAX_SEPARATION_DEFAULT = 80

def generate_stereogram_from_pil(depth_map_pil, texture_path, output_path, min_sep, max_sep):
    """
    Generates a single image stereogram (autostereogram) from a PIL Image depth map.

    Args:
        depth_map_pil (PIL.Image.Image): Grayscale depth map PIL Image.
                                         White/light = near, black/dark = far.
        texture_path (str): Path to the texture image to tile.
        output_path (str): Path to save the generated stereogram.
        min_sep (int): Minimum separation/period for the furthest points (pixels).
        max_sep (int): Maximum separation/period for the closest points (pixels).
    """
    try:
        # Ensure depth map is grayscale for processing
        depth_map_img = depth_map_pil.convert('L')
        texture_img = Image.open(texture_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Texture image '{texture_path}' not found.")
        return
    except Exception as e:
        print(f"Error opening images: {e}")
        return

    width, height = depth_map_img.size
    texture_width, texture_height = texture_img.size

    stereogram_img = Image.new('RGB', (width, height))

    depth_pixels = depth_map_img.load()
    texture_pixels = texture_img.load()
    output_pixels = stereogram_img.load()

    print(f"Generating stereogram ({width}x{height})...")

    for y in range(height):
        for x in range(width):
            depth_value_normalized = depth_pixels[x, y] / 255.0
            current_separation = int(min_sep + (max_sep - min_sep) * depth_value_normalized)

            if x < current_separation:
                tx = x % texture_width
                ty = y % texture_height
                output_pixels[x, y] = texture_pixels[tx, ty]
            else:
                # Ensure we don't read from a negative index if current_separation is too large
                # for x, though this shouldn't happen if current_separation is reasonable.
                ref_x = x - current_separation
                if ref_x < 0: # Should ideally be caught by x < current_separation
                    tx = x % texture_width
                    ty = y % texture_height
                    output_pixels[x, y] = texture_pixels[tx, ty]
                else:
                    output_pixels[x, y] = output_pixels[ref_x, y]
        
        if (y + 1) % 50 == 0 or (y + 1) == height:
            print(f"Processed stereogram row {y + 1}/{height}")

    try:
        stereogram_img.save(output_path)
        print(f"Stereogram saved to {output_path}")
    except Exception as e:
        print(f"Error saving output stereogram: {e}")

def create_depth_map_from_image(image_path, model_type="MiDaS_small"):
    """
    Creates a depth map from an input image using MiDaS.

    Args:
        image_path (str): Path to the input color image.
        model_type (str): Type of MiDaS model to use.
                          ("MiDaS_small", "DPT_Large", "DPT_Hybrid")

    Returns:
        PIL.Image.Image: Grayscale depth map (white=near, black=far), or None on error.
    """
    print(f"Loading MiDaS model ({model_type})...")
    try:
        # Load MiDaS model from PyTorch Hub
        # model_type = "MiDaS_small"  # Fast, good for general use
        # model_type = "DPT_Large"   # More accurate, slower
        # model_type = "DPT_Hybrid"  # A balance
        model = torch.hub.load("intel-isl/MiDaS", model_type)

        # Determine device
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        model.eval()
        print(f"Using device: {device}")

        # Load appropriate transform
        transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform if "DPT" in model_type \
            else torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

        print(f"Loading and transforming input image: {image_path}")
        img_pil = Image.open(image_path).convert("RGB")
        img_cv = np.array(img_pil) # PIL to OpenCV
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR) # MiDaS example uses BGR

        input_batch = transform(img_cv).to(device)

        with torch.no_grad():
            print("Predicting depth...")
            prediction = model(input_batch)

            # Resize prediction to original image size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_pil.size[::-1], # (height, width) for interpolate
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_output = prediction.cpu().numpy()

        # Normalize the depth map to 0-255 and invert (MiDaS output is often inverse: higher value = further)
        # We want: higher value (whiter) = closer for our stereogram function
        depth_normalized = (depth_output - np.min(depth_output)) / (np.max(depth_output) - np.min(depth_output))
        depth_inverted_normalized = 1.0 - depth_normalized # Invert: 0=far, 1=near
        
        depth_map_visual = (depth_inverted_normalized * 255).astype(np.uint8)
        depth_map_pil = Image.fromarray(depth_map_visual)

        print("Depth map generated successfully.")
        return depth_map_pil

    except Exception as e:
        print(f"Error during depth map generation: {e}")
        print("Please ensure you have an internet connection for the first run to download models.")
        print("Also, ensure PyTorch, torchvision, timm, and opencv-python are installed correctly.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate an autostereogram using AI depth estimation.")
    parser.add_argument("--input", required=True, help="Path to the input color image for depth estimation.")
    parser.add_argument("--texture", required=True, help="Path to the input texture image.")
    parser.add_argument("--output", required=True, help="Path for the output stereogram image.")
    parser.add_argument("--midasmodel", default="MiDaS_small", choices=["MiDaS_small", "DPT_Large", "DPT_Hybrid"],
                        help="Which MiDaS model to use for depth estimation.")
    parser.add_argument("--minsep", type=int, default=MIN_SEPARATION_DEFAULT, help="Min separation for far points (pixels).")
    parser.add_argument("--maxsep", type=int, default=MAX_SEPARATION_DEFAULT, help="Max separation for near points (pixels).")
    parser.add_argument("--save_depthmap", type=str, default=None, help="Optional: Path to save the generated grayscale depth map.")


    args = parser.parse_args()

    if args.minsep >= args.maxsep:
        print("Error: min_separation must be less than max_separation.")
        return

    # 1. Create depth map from input image
    generated_depth_map_pil = create_depth_map_from_image(args.input, model_type=args.midasmodel)

    if generated_depth_map_pil:
        if args.save_depthmap:
            try:
                generated_depth_map_pil.save(args.save_depthmap)
                print(f"Generated depth map saved to {args.save_depthmap}")
            except Exception as e:
                print(f"Error saving generated depth map: {e}")

        # 2. Generate stereogram using the AI-generated depth map
        generate_stereogram_from_pil(
            generated_depth_map_pil,
            args.texture,
            args.output,
            args.minsep,
            args.maxsep
        )
    else:
        print("Could not generate depth map. Exiting.")


if __name__ == "__main__":
    main()
