import argparse
from PIL import Image, ImageOps
import torch
import numpy as np
import cv2 # OpenCV for transformations
import os # For path operations
from datetime import datetime # For unique output names if needed

# Import the refactored texture generation module
import deeptexture 

# Constants for stereogram generation (can be args later)
MIN_SEPARATION_DEFAULT = 70
MAX_SEPARATION_DEFAULT = 80

# --- Stereogram Generation Function ---
# IMPORTANT: Modified to accept texture_pil (PIL.Image) instead of texture_path
def generate_stereogram_from_pil_texture(depth_map_pil, texture_pil, output_path, min_sep, max_sep):
    """
    Generates a single image stereogram (autostereogram) from a PIL Image depth map
    and a PIL Image texture.
    """
    try:
        depth_map_img = depth_map_pil.convert('L')
        # Texture is already a PIL image, ensure it's RGB for consistency
        texture_img = texture_pil.convert('RGB')
    except Exception as e:
        print(f"Error preparing images for stereogram: {e}")
        return False

    width, height = depth_map_img.size
    texture_width, texture_height = texture_img.size

    stereogram_img = Image.new('RGB', (width, height))

    depth_pixels = depth_map_img.load()
    texture_pixels = texture_img.load()
    output_pixels = stereogram_img.load()

    print(f"Generating stereogram ({width}x{height})...")

    for y in tqdm(range(height), desc="Stereogram Rows", leave=False):
        for x in range(width):
            depth_value_normalized = depth_pixels[x, y] / 255.0
            current_separation = int(min_sep + (max_sep - min_sep) * depth_value_normalized)
            current_separation = max(1, current_separation) 

            if x < current_separation:
                tx = x % texture_width
                ty = y % texture_height
                output_pixels[x, y] = texture_pixels[tx, ty]
            else:
                ref_x = x - current_separation
                output_pixels[x, y] = output_pixels[ref_x, y]
    
    try:
        stereogram_img.save(output_path)
        # print(f"Stereogram saved to {output_path}") # Main will print final summary
        return True
    except Exception as e:
        print(f"Error saving output stereogram: {e}")
        return False

# --- Depth Map Generation Function ---
def create_depth_map_from_image(image_path, model_type="MiDaS_small", target_size=None, verbose=True):
    """
    Creates a depth map from an input image using MiDaS.
    Returns a PIL Image (depth map) and the PIL image that was processed by MiDaS (potentially resized).
    """
    if verbose: print(f"Depth Map Gen: Loading MiDaS model ({model_type})...")
    try:
        # Forcing trust_repo for transforms as it's sometimes needed
        model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        transform_hub_path = "intel-isl/MiDaS" 
        transform_name = "dpt_transform" if "DPT" in model_type else "small_transform"
        transform = torch.hub.load(transform_hub_path, "transforms", trust_repo=True)[transform_name]

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device); model.eval()
        if verbose: print(f"Depth Map Gen: Using device: {device}")

        if verbose: print(f"Depth Map Gen: Loading and transforming input image: {image_path}")
        img_pil_orig = Image.open(image_path).convert("RGB")
        img_for_midas_processing = img_pil_orig.copy() # Start with a copy

        if target_size:
            if verbose: print(f"Depth Map Gen: Resizing input for MiDaS to {target_size}...")
            img_for_midas_processing = img_pil_orig.resize(target_size, Image.Resampling.LANCZOS)
        
        img_cv = np.array(img_for_midas_processing); img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        input_batch = transform(img_cv).to(device)

        with torch.no_grad():
            if verbose: print("Depth Map Gen: Predicting depth...")
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=img_for_midas_processing.size[::-1],
                mode="bicubic", align_corners=False
            ).squeeze()
        depth_output = prediction.cpu().numpy()
        depth_min, depth_max = np.min(depth_output), np.max(depth_output)
        depth_normalized = (depth_output - depth_min) / (depth_max - depth_min) if depth_max > depth_min else np.zeros_like(depth_output)
        depth_inverted_normalized = 1.0 - depth_normalized
        depth_map_visual = (depth_inverted_normalized * 255).astype(np.uint8)
        depth_map_pil = Image.fromarray(depth_map_visual)

        # Resize depth map back to original dimensions of the *initial input*
        if depth_map_pil.size != img_pil_orig.size:
            if verbose: print(f"Depth Map Gen: Resizing depth map from {depth_map_pil.size} back to original image size: {img_pil_orig.size}")
            depth_map_pil = depth_map_pil.resize(img_pil_orig.size, Image.Resampling.LANCZOS)
        
        if verbose: print("Depth Map Gen: Depth map generated successfully.")
        return depth_map_pil, img_pil_orig # Return original for texture base consistency if needed
    except Exception as e:
        print(f"Error during depth map generation: {e}"); return None, None


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="DeepStereo: AI-Powered Autostereogram Generator.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Stereogram Args
    stereo_group = parser.add_argument_group('Stereogram Generation Parameters')
    stereo_group.add_argument("--input", required=True, help="Path to the input color image (for depth and optionally texture base).")
    stereo_group.add_argument("--output", required=True, help="Path for the output stereogram image.")
    stereo_group.add_argument("--minsep", type=int, default=MIN_SEPARATION_DEFAULT, help="Min separation for far points (pixels).")
    stereo_group.add_argument("--maxsep", type=int, default=MAX_SEPARATION_DEFAULT, help="Max separation for near points (pixels).")

    # Depth Map Args
    depth_group = parser.add_argument_group('Depth Map Generation (MiDaS)')
    depth_group.add_argument("--midasmodel", default="MiDaS_small", choices=["MiDaS_small", "DPT_Large", "DPT_Hybrid"], help="MiDaS model for depth estimation.")
    depth_group.add_argument("--save_depthmap", type=str, default=None, help="Optional: Path to save the AI-generated grayscale depth map.")
    depth_group.add_argument("--depth_model_input_width", type=int, default=384, help="Width to resize input image to before MiDaS (maintains aspect, rounded to 32px). 0 for no resize.")

    # Texture Args
    texture_source_group = parser.add_argument_group('Texture Source')
    texture_source_group.add_argument("--texture", default=None, help="Path to an external texture image. If not provided, --generate_texture_on_the_fly must be used or a default random texture is created.")
    texture_source_group.add_argument("--generate_texture_on_the_fly", action="store_true", help="Generate texture on-the-fly using the main input image as a base. Overrides --texture if both specified.")
    texture_source_group.add_argument("--save_generated_texture", type=str, default=None, help="Optional: Path to save the on-the-fly generated texture image.")


    # On-the-fly Texture Generation Args (from deeptexture.py, prefixed with tex_)
    tex_gen_group = parser.add_argument_group('On-the-fly Texture Generation Parameters (used if --generate_texture_on_the_fly)')
    tex_gen_group.add_argument("--tex_max_megapixels", type=float, default=1.0, help="Resize base image for texture to approx this MP (default: 1.0). 0 for no resize.")
    tex_gen_group.add_argument("--tex_combination_mode", choices=["sequential", "blend"], default="sequential", help="Texture: How to combine method outputs.")
    tex_gen_group.add_argument("--tex_blend_type", choices=["average", "lighten", "darken", "multiply", "screen", "add", "difference", "overlay"], default="overlay", help="Texture: Blend mode.")
    tex_gen_group.add_argument("--tex_blend_opacity", type=float, default=1.0, help="Texture: Blend opacity (0.0-1.0).")
    
    # Method 1 Args
    tex_m1_group = tex_gen_group.add_argument_group('Texture Method 1: Color Dots')
    tex_m1_group.add_argument("--tex_method1_color_dots", action="store_true", help="Texture M1: Apply.")
    tex_m1_group.add_argument("--tex_m1_density", type=float, default=0.7, help="Texture M1: Dot density.")
    tex_m1_group.add_argument("--tex_m1_dot_size", type=int, default=2, help="Texture M1: Dot size.")
    tex_m1_group.add_argument("--tex_m1_bg_color", type=str, default="black", help="Texture M1: BG color.")
    tex_m1_group.add_argument("--tex_m1_color_mode", choices=["content_pixel", "random_rgb", "random_from_palette", "transformed_hue", "transformed_invert"], default="content_pixel", help="Texture M1: Color mode.")
    tex_m1_group.add_argument("--tex_m1_hue_shift_degrees", type=float, default=90, help="Texture M1: Hue shift.")
    
    # Method 2 Args
    tex_m2_group = tex_gen_group.add_argument_group('Texture Method 2: Density/Size Driven')
    tex_m2_group.add_argument("--tex_method2_density_size", action="store_true", help="Texture M2: Apply.")
    tex_m2_group.add_argument("--tex_m2_mode", choices=["density", "size"], default="density", help="Texture M2: Mode.")
    tex_m2_group.add_argument("--tex_m2_element_color", type=str, default="white", help="Texture M2: Element color.")
    tex_m2_group.add_argument("--tex_m2_bg_color", type=str, default="black", help="Texture M2: BG color.")
    tex_m2_group.add_argument("--tex_m2_base_size", type=int, default=3, help="Texture M2: Base size.")
    tex_m2_group.add_argument("--tex_m2_max_size", type=int, default=12, help="Texture M2: Max size.")
    tex_m2_group.add_argument("--tex_m2_invert_influence", action="store_true", help="Texture M2: Invert influence.")
    tex_m2_group.add_argument("--tex_m2_density_factor", type=float, default=1.0, help="Texture M2: Density factor.")

    # Method 3 Args
    tex_m3_group = tex_gen_group.add_argument_group('Texture Method 3: Voronoi')
    tex_m3_group.add_argument("--tex_method3_voronoi", action="store_true", help="Texture M3: Apply.")
    tex_m3_group.add_argument("--tex_m3_num_points", type=int, default=200, help="Texture M3: Num points.")
    tex_m3_group.add_argument("--tex_m3_metric", choices=["F1", "F2", "F2-F1"], default="F1", help="Texture M3: Metric.")
    tex_m3_group.add_argument("--tex_m3_color_source", choices=["distance", "content_point_color", "voronoi_cell_content_color"], default="distance", help="Texture M3: Color source.")

    # Method 4 Args
    tex_m4_group = tex_gen_group.add_argument_group('Texture Method 4: Glyph Dither')
    tex_m4_group.add_argument("--tex_method4_glyph_dither", action="store_true", help="Texture M4: Apply.")
    tex_m4_group.add_argument("--tex_m4_num_colors", type=int, default=8, help="Texture M4: Num colors for quantization.")
    tex_m4_group.add_argument("--tex_m4_glyph_size", type=int, default=10, help="Texture M4: Glyph size.")
    tex_m4_group.add_argument("--tex_m4_glyph_style", choices=["random_dots", "lines", "circles", "solid"], default="random_dots", help="Texture M4: Glyph style.")
    tex_m4_group.add_argument("--tex_m4_use_quantized_color_for_glyph_element", action="store_true", help="Texture M4: Use quantized color for glyph elements.")

    args = parser.parse_args()

    print("--- DeepStereo Generator ---")

    if args.minsep >= args.maxsep: print("Error: --minsep must be less than --maxsep."); return

    # Determine target size for MiDaS model input
    target_midas_processing_size = None
    if args.depth_model_input_width and args.depth_model_input_width > 0:
        try:
            with Image.open(args.input) as temp_img: # Ensure file is closed
                orig_w, orig_h = temp_img.size
            aspect_ratio = orig_h / orig_w
            target_w_midas = (args.depth_model_input_width // 32) * 32
            target_h_midas = (int(target_w_midas * aspect_ratio) // 32) * 32
            if target_w_midas > 0 and target_h_midas > 0:
                target_midas_processing_size = (target_w_midas, target_h_midas)
            else: print(f"Warning: Calculated MiDaS processing size invalid. Using original size for depth map.")
        except Exception as e: print(f"Warning: Could not get input image size for MiDaS resize. Error: {e}. Using original size for depth map.")
    
    # 1. Create depth map from input image
    # create_depth_map_from_image now returns (depth_map_pil, original_input_pil_for_texture_base)
    generated_depth_map_pil, original_input_pil_for_texture_base = create_depth_map_from_image(
                                                                        args.input, 
                                                                        model_type=args.midasmodel,
                                                                        target_size=target_midas_processing_size
                                                                    )

    if not generated_depth_map_pil: print("Could not generate depth map. Exiting."); return
    if args.save_depthmap:
        try: generated_depth_map_pil.save(args.save_depthmap); print(f"Generated depth map saved to {args.save_depthmap}")
        except Exception as e: print(f"Error saving generated depth map: {e}")

    # 2. Prepare Texture
    texture_to_use_pil = None
    texture_base_image_pil = None # This will be the image used as input for deeptexture

    if args.generate_texture_on_the_fly:
        print("Preparing for on-the-fly texture generation...")
        if original_input_pil_for_texture_base:
            texture_base_image_pil = original_input_pil_for_texture_base.copy() # Use the original input
        else: # Should not happen if depth map gen succeeded
            try: texture_base_image_pil = Image.open(args.input).convert("RGB") 
            except Exception as e: print(f"Error loading base image for texture gen: {e}"); texture_base_image_pil = None
        
        if texture_base_image_pil:
            # Pass the 'args' object directly to deeptexture's core function
            # It will know to look for 'tex_' prefixed attributes
            texture_to_use_pil = deeptexture.generate_texture_from_config(
                texture_base_image_pil,
                args, # Pass the main args object from deepstereo
                verbose=True
            )
            if args.save_generated_texture and texture_to_use_pil:
                try: texture_to_use_pil.save(args.save_generated_texture); print(f"Saved on-the-fly generated texture to {args.save_generated_texture}")
                except Exception as e: print(f"Error saving generated texture: {e}")
        else:
            print("Could not prepare base image for on-the-fly texture generation.")

    if texture_to_use_pil is None: # If not generated, or generation failed
        if args.texture:
            try:
                print(f"Loading texture from file: {args.texture}")
                texture_to_use_pil = Image.open(args.texture).convert('RGB')
            except FileNotFoundError: print(f"Error: Texture file '{args.texture}' not found."); return
            except Exception as e: print(f"Error opening texture file: {e}"); return
        else:
            print("Error: No texture source. Provide --texture or use --generate_texture_on_the_fly with methods.")
            print("Creating a default random noise texture as fallback.")
            # Use depth map size for default texture, or a fixed size if depth map also failed (though we exit earlier)
            w_fallback, h_fallback = generated_depth_map_pil.size if generated_depth_map_pil else (512,512)
            noise_data = np.random.randint(0, 256, (h_fallback, w_fallback, 3), dtype=np.uint8)
            texture_to_use_pil = Image.fromarray(noise_data)

    if not texture_to_use_pil: print("Critical error: Texture could not be loaded or generated. Exiting."); return

    # 3. Generate Stereogram
    print("Proceeding to stereogram generation...")
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory for stereogram: {output_dir}")

    success = generate_stereogram_from_pil_texture( # Use the renamed function
        generated_depth_map_pil,
        texture_to_use_pil,
        args.output,
        args.minsep,
        args.maxsep
    )

    if success: print(f"DeepStereo generation complete! Output: {args.output}")
    else: print("DeepStereo generation failed.")

if __name__ == "__main__":
    main()
