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
        return False
    except Exception as e:
        print(f"Error opening images: {e}")
        return False

    width, height = depth_map_img.size
    texture_width, texture_height = texture_img.size

    stereogram_img = Image.new('RGB', (width, height))

    depth_pixels = depth_map_img.load()
    texture_pixels = texture_img.load()
    output_pixels = stereogram_img.load()

    print(f"Generating stereogram ({width}x{height})...")

    for y in range(height):
        for x in range(width):
            # Get normalized depth (0.0 = far/black, 1.0 = near/white from depth_map_pil)
            depth_value_normalized = depth_pixels[x, y] / 255.0

            # Calculate the required separation for this pixel's depth
            # Closer points (depth_value_normalized = 1.0) get max_sep
            # Further points (depth_value_normalized = 0.0) get min_sep
            current_separation = int(min_sep + (max_sep - min_sep) * depth_value_normalized)
            current_separation = max(1, current_separation) # Ensure separation is at least 1

            if x < current_separation:
                # Not enough space to the left. Fill with texture.
                tx = x % texture_width
                ty = y % texture_height
                output_pixels[x, y] = texture_pixels[tx, ty]
            else:
                # Copy color from the pixel 'current_separation' to the left
                ref_x = x - current_separation
                # ref_x should be >= 0 due to the check above.
                output_pixels[x, y] = output_pixels[ref_x, y]
        
        if (y + 1) % 50 == 0 or (y + 1) == height: # Print progress
            print(f"Processed stereogram row {y + 1}/{height}")

    try:
        stereogram_img.save(output_path)
        print(f"Stereogram saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving output stereogram: {e}")
        return False

def create_depth_map_from_image(image_path, model_type="MiDaS_small", target_size=None):
    """
    Creates a depth map from an input image using MiDaS.

    Args:
        image_path (str): Path to the input color image.
        model_type (str): Type of MiDaS model to use.
                          ("MiDaS_small", "DPT_Large", "DPT_Hybrid")
        target_size (tuple, optional): (width, height) to resize the input image before processing.
                                       None means use original size.

    Returns:
        PIL.Image.Image: Grayscale depth map (white=near, black=far), or None on error.
    """
    print(f"Loading MiDaS model ({model_type})...")
    try:
        model = torch.hub.load("intel-isl/MiDaS", model_type)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        model.eval()
        print(f"Using device: {device}")

        transform_name = "dpt_transform" if "DPT" in model_type else "small_transform"
        transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)[transform_name]


        print(f"Loading and transforming input image: {image_path}")
        img_pil_orig = Image.open(image_path).convert("RGB")

        # Resize if target_size is specified
        if target_size:
            print(f"Resizing input image to {target_size} for depth estimation...")
            img_pil_transformed = img_pil_orig.resize(target_size, Image.Resampling.LANCZOS)
        else:
            img_pil_transformed = img_pil_orig
        
        img_cv = np.array(img_pil_transformed)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR) # MiDaS example uses BGR

        input_batch = transform(img_cv).to(device)

        with torch.no_grad():
            print("Predicting depth...")
            prediction = model(input_batch)

            # Resize prediction to original image size (or target_size if that was used for processing)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_pil_transformed.size[::-1], # (height, width) for interpolate
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_output = prediction.cpu().numpy()

        # Normalize the depth map: 0 for far, 1 for near
        depth_min = np.min(depth_output)
        depth_max = np.max(depth_output)

        if depth_max == depth_min: # Avoid division by zero if depth is flat
            depth_normalized = np.zeros_like(depth_output)
        else:
            depth_normalized = (depth_output - depth_min) / (depth_max - depth_min)
        
        # Invert: MiDaS typically outputs higher values for further away. We want higher for nearer.
        # So, after normalization (0=min_depth, 1=max_depth from model),
        # if min_depth was further, then 0 is far, 1 is near. This might be what we want.
        # If MiDaS raw: large value = far, small value = near
        # Normalized: 0 = near, 1 = far
        # Inverted: 1-normalized => 1 = near, 0 = far. This matches our stereogram logic.
        depth_inverted_normalized = 1.0 - depth_normalized
        
        depth_map_visual = (depth_inverted_normalized * 255).astype(np.uint8)
        depth_map_pil = Image.fromarray(depth_map_visual)

        # If the original input was resized for the model, resize the depth map back to original dimensions
        if target_size and target_size != img_pil_orig.size:
            print(f"Resizing depth map back to original image size: {img_pil_orig.size}")
            depth_map_pil = depth_map_pil.resize(img_pil_orig.size, Image.Resampling.LANCZOS)


        print("Depth map generated successfully.")
        return depth_map_pil

    except Exception as e:
        print(f"Error during depth map generation: {e}")
        print("Please ensure you have an internet connection for the first run to download models.")
        print("Ensure PyTorch, torchvision, timm, and opencv-python are installed correctly.")
        return None

def main():
    parser = argparse.ArgumentParser(description="DeepStereo: AI-Powered Autostereogram Generator.")
    parser.add_argument("--input", required=True, help="Path to the input color image for depth estimation.")
    parser.add_argument("--texture", required=True, help="Path to the input texture image.")
    parser.add_argument("--output", required=True, help="Path for the output stereogram image.")
    parser.add_argument("--midasmodel", default="MiDaS_small", choices=["MiDaS_small", "DPT_Large", "DPT_Hybrid"],
                        help="Which MiDaS model to use for depth estimation. (default: MiDaS_small)")
    parser.add_argument("--minsep", type=int, default=MIN_SEPARATION_DEFAULT, 
                        help=f"Min separation for far points (pixels). (default: {MIN_SEPARATION_DEFAULT})")
    parser.add_argument("--maxsep", type=int, default=MAX_SEPARATION_DEFAULT, 
                        help=f"Max separation for near points (pixels). (default: {MAX_SEPARATION_DEFAULT})")
    parser.add_argument("--save_depthmap", type=str, default=None, 
                        help="Optional: Path to save the generated grayscale depth map.")
    parser.add_argument("--depth_model_input_width", type=int, default=None,
                        help="Optional: Width to resize input image to before MiDaS processing (e.g., 384, 512). Aspect ratio maintained.")

    args = parser.parse_args()

    if args.minsep >= args.maxsep:
        print("Error: --minsep must be less than --maxsep.")
        return

    # Determine target size for MiDaS model input if width is specified
    target_processing_size = None
    if args.depth_model_input_width:
        try:
            temp_img = Image.open(args.input)
            orig_w, orig_h = temp_img.size
            aspect_ratio = orig_h / orig_w
            target_h = int(args.depth_model_input_width * aspect_ratio)
            # MiDaS models often prefer inputs that are multiples of 32
            target_w = (args.depth_model_input_width // 32) * 32
            target_h = (target_h // 32) * 32
            if target_w > 0 and target_h > 0:
                 target_processing_size = (target_w, target_h)
            else:
                print(f"Warning: Calculated target processing size ({target_w}x{target_h}) is too small based on --depth_model_input_width. Using original size.")
            temp_img.close()
        except Exception as e:
            print(f"Warning: Could not determine original image size for aspect ratio. Error: {e}. Using original size for MiDaS.")
            target_processing_size = None


    # 1. Create depth map from input image
    generated_depth_map_pil = create_depth_map_from_image(args.input, 
                                                          model_type=args.midasmodel,
                                                          target_size=target_processing_size)

    if generated_depth_map_pil:
        if args.save_depthmap:
            try:
                generated_depth_map_pil.save(args.save_depthmap)
                print(f"Generated depth map saved to {args.save_depthmap}")
            except Exception as e:
                print(f"Error saving generated depth map: {e}")

        # 2. Generate stereogram using the AI-generated depth map
        success = generate_stereogram_from_pil(
            generated_depth_map_pil,
            args.texture,
            args.output,
            args.minsep,
            args.maxsep
        )
        if success:
            print("DeepStereo generation complete!")
        else:
            print("DeepStereo generation failed.")
    else:
        print("Could not generate depth map. Exiting.")


if __name__ == "__main__":
    main()
