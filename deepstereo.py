import argparse
from PIL import Image, ImageOps
import torch
import numpy as np
import cv2 # OpenCV for transformations
import os # For path operations
from datetime import datetime # For unique output names
from tqdm import tqdm # Ensure tqdm is imported here

# Import the refactored texture generation module
import deeptexture 

# Constants for stereogram generation (can be args later)
# Let's try a wider range for more depth "resolution"
MIN_SEPARATION_DEFAULT = 20  # Smaller value for furthest points
MAX_SEPARATION_DEFAULT = 200 # Larger value for closest, giving a range of 40 pixels

# --- Stereogram Generation Function ---
# (generate_stereogram_from_pil_texture - no change needed here from last version, tqdm was added to its loop)
def generate_stereogram_from_pil_texture(depth_map_pil, texture_pil, output_path, min_sep, max_sep):
    try:
        depth_map_img = depth_map_pil.convert('L')
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

    # tqdm progress bar for stereogram generation rows
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
        return True
    except Exception as e:
        print(f"Error saving output stereogram: {e}")
        return False

# --- Depth Map Generation Function ---
# (create_depth_map_from_image - no change needed here from last version)
def create_depth_map_from_image(image_path, model_type="MiDaS_small", target_size=None, verbose=True):
    if verbose: print(f"Depth Map Gen: Loading MiDaS model ({model_type})...")
    try:
        model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid": transform = midas_transforms.dpt_transform
        elif model_type == "MiDaS_small": transform = midas_transforms.small_transform
        else: print(f"Warning: Unknown MiDaS model type '{model_type}'. Using small_transform."); transform = midas_transforms.small_transform
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device); model.eval()
        if verbose: print(f"Depth Map Gen: Using device: {device}")
        if verbose: print(f"Depth Map Gen: Loading and transforming input image: {image_path}")
        img_pil_orig = Image.open(image_path).convert("RGB")
        img_for_midas_processing = img_pil_orig.copy()
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
        if depth_map_pil.size != img_pil_orig.size:
            if verbose: print(f"Depth Map Gen: Resizing depth map from {depth_map_pil.size} back to original image size: {img_pil_orig.size}")
            depth_map_pil = depth_map_pil.resize(img_pil_orig.size, Image.Resampling.LANCZOS)
        if verbose: print("Depth Map Gen: Depth map generated successfully.")
        return depth_map_pil, img_pil_orig 
    except Exception as e:
        print(f"Error during depth map generation: {e}"); return None, None


# --- Main Execution ---
# (main function - no changes needed to its argument parsing or overall structure from the previous version,
#  as the MIN/MAX_SEPARATION_DEFAULTs are what we're adjusting directly above)
def main():
    parser = argparse.ArgumentParser(description="DeepStereo: AI-Powered Autostereogram Generator.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    stereo_group = parser.add_argument_group('Stereogram Generation Parameters')
    stereo_group.add_argument("--input", required=True, help="Path to the input color image (for depth and optionally texture base).")
    stereo_group.add_argument("--output_dir", default="output_stereograms", help="Directory to save the generated stereogram image.")
    stereo_group.add_argument("--output_filename_base", default=None, help="Optional base for output filename. If None, uses input filename base.")
    stereo_group.add_argument("--minsep", type=int, default=MIN_SEPARATION_DEFAULT, help="Min separation for far points (pixels).")
    stereo_group.add_argument("--maxsep", type=int, default=MAX_SEPARATION_DEFAULT, help="Max separation for near points (pixels).")

    depth_group = parser.add_argument_group('Depth Map Generation (MiDaS)')
    depth_group.add_argument("--midasmodel", default="MiDaS_small", choices=["MiDaS_small", "DPT_Large", "DPT_Hybrid"], help="MiDaS model for depth estimation.")
    depth_group.add_argument("--save_depthmap", type=str, default=None, help="Optional: Path to save the AI-generated grayscale depth map (full path including filename).")
    depth_group.add_argument("--depth_model_input_width", type=int, default=384, help="Width to resize input image to before MiDaS (maintains aspect, rounded to 32px). 0 for no resize.")

    texture_source_group = parser.add_argument_group('Texture Source')
    texture_source_group.add_argument("--texture", default=None, help="Path to an external texture image. If not provided or overridden, --generate_texture_on_the_fly is used.")
    texture_source_group.add_argument("--generate_texture_on_the_fly", action="store_true", help="Generate texture on-the-fly using the main input image as a base. Overrides --texture if specified.")
    texture_source_group.add_argument("--save_generated_texture", type=str, default=None, help="Optional: Path to save the on-the-fly generated texture image (full path including filename).")

    tex_gen_group = parser.add_argument_group('On-the-fly Texture Generation Parameters (used if --generate_texture_on_the_fly)')
    tex_gen_group.add_argument("--tex_max_megapixels", type=float, default=1.0, help="Resize base image for texture to approx this MP. 0 for no resize.")
    tex_gen_group.add_argument("--tex_combination_mode", choices=["sequential", "blend"], default="sequential", help="Texture: How to combine method outputs.")
    tex_gen_group.add_argument("--tex_blend_type", choices=["average", "lighten", "darken", "multiply", "screen", "add", "difference", "overlay"], default="overlay", help="Texture: Blend mode.")
    tex_gen_group.add_argument("--tex_blend_opacity", type=float, default=1.0, help="Texture: Blend opacity (0.0-1.0).")
    
    tex_m1_group = tex_gen_group.add_argument_group('Texture Method 1: Color Dots')
    tex_m1_group.add_argument("--tex_method1_color_dots", action="store_true", help="Texture M1: Apply.")
    tex_m1_group.add_argument("--tex_m1_density", type=float, default=0.7, help="Texture M1: Dot density.")
    tex_m1_group.add_argument("--tex_m1_dot_size", type=int, default=2, help="Texture M1: Dot size.")
    tex_m1_group.add_argument("--tex_m1_bg_color", type=str, default="black", help="Texture M1: BG color.")
    tex_m1_group.add_argument("--tex_m1_color_mode", choices=["content_pixel", "random_rgb", "random_from_palette", "transformed_hue", "transformed_invert"], default="content_pixel", help="Texture M1: Color mode.")
    tex_m1_group.add_argument("--tex_m1_hue_shift_degrees", type=float, default=90, help="Texture M1: Hue shift.")
    
    tex_m2_group = tex_gen_group.add_argument_group('Texture Method 2: Density/Size Driven')
    tex_m2_group.add_argument("--tex_method2_density_size", action="store_true", help="Texture M2: Apply.")
    tex_m2_group.add_argument("--tex_m2_mode", choices=["density", "size"], default="density", help="Texture M2: Mode.")
    tex_m2_group.add_argument("--tex_m2_element_color", type=str, default="white", help="Texture M2: Element color.")
    tex_m2_group.add_argument("--tex_m2_bg_color", type=str, default="black", help="Texture M2: BG color.")
    tex_m2_group.add_argument("--tex_m2_base_size", type=int, default=3, help="Texture M2: Base size.")
    tex_m2_group.add_argument("--tex_m2_max_size", type=int, default=12, help="Texture M2: Max size.")
    tex_m2_group.add_argument("--tex_m2_invert_influence", action="store_true", help="Texture M2: Invert influence.")
    tex_m2_group.add_argument("--tex_m2_density_factor", type=float, default=1.0, help="Texture M2: Density factor.")

    tex_m3_group = tex_gen_group.add_argument_group('Texture Method 3: Voronoi')
    tex_m3_group.add_argument("--tex_method3_voronoi", action="store_true", help="Texture M3: Apply.")
    tex_m3_group.add_argument("--tex_m3_num_points", type=int, default=200, help="Texture M3: Num points.")
    tex_m3_group.add_argument("--tex_m3_metric", choices=["F1", "F2", "F2-F1"], default="F1", help="Texture M3: Metric.")
    tex_m3_group.add_argument("--tex_m3_color_source", choices=["distance", "content_point_color", "voronoi_cell_content_color"], default="distance", help="Texture M3: Color source.")

    tex_m4_group = tex_gen_group.add_argument_group('Texture Method 4: Glyph Dither')
    tex_m4_group.add_argument("--tex_method4_glyph_dither", action="store_true", help="Texture M4: Apply.")
    tex_m4_group.add_argument("--tex_m4_num_colors", type=int, default=8, help="Texture M4: Num colors for quantization.")
    tex_m4_group.add_argument("--tex_m4_glyph_size", type=int, default=10, help="Texture M4: Glyph size.")
    tex_m4_group.add_argument("--tex_m4_glyph_style", choices=["random_dots", "lines", "circles", "solid"], default="random_dots", help="Texture M4: Glyph style.")
    tex_m4_group.add_argument("--tex_m4_use_quantized_color_for_glyph_element", action="store_true", help="Texture M4: Use quantized color for glyph elements.")

    args = parser.parse_args()
    print("--- DeepStereo Generator ---")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")

    if args.minsep >= args.maxsep: print("Error: --minsep must be less than --maxsep."); return

    input_filename_base = os.path.splitext(os.path.basename(args.input))[0]
    output_filename_base = args.output_filename_base if args.output_filename_base else input_filename_base
    filename_suffix = ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    target_midas_processing_size = None
    if args.depth_model_input_width and args.depth_model_input_width > 0:
        try:
            with Image.open(args.input) as temp_img: orig_w, orig_h = temp_img.size
            aspect_ratio = orig_h / orig_w
            target_w_midas = (args.depth_model_input_width // 32) * 32
            target_h_midas = (int(target_w_midas * aspect_ratio) // 32) * 32
            if target_w_midas > 0 and target_h_midas > 0: target_midas_processing_size = (target_w_midas, target_h_midas)
            else: print(f"Warning: Calculated MiDaS processing size invalid. Using original size for depth map.")
        except Exception as e: print(f"Warning: Could not get input image size for MiDaS resize. Error: {e}.")
    
    generated_depth_map_pil, original_input_pil_for_texture_base = create_depth_map_from_image(
        args.input, model_type=args.midasmodel, target_size=target_midas_processing_size
    )

    if not generated_depth_map_pil: print("Could not generate depth map. Exiting."); return
    filename_suffix += f"_{args.midasmodel}"
    if target_midas_processing_size: filename_suffix += f"_depth{args.depth_model_input_width}w"

    if args.save_depthmap:
        depthmap_save_path = args.save_depthmap
        if os.path.isdir(args.save_depthmap) or not os.path.splitext(args.save_depthmap)[1]:
             depthmap_filename = f"{output_filename_base}{filename_suffix}_depthmap_{timestamp}.png"
             depthmap_save_path = os.path.join(args.save_depthmap if os.path.isdir(args.save_depthmap) else args.output_dir, depthmap_filename)
        try: 
            os.makedirs(os.path.dirname(depthmap_save_path), exist_ok=True)
            generated_depth_map_pil.save(depthmap_save_path); print(f"Generated depth map saved to {depthmap_save_path}")
        except Exception as e: print(f"Error saving generated depth map: {e}")

    texture_to_use_pil = None
    texture_base_image_pil = None
    texture_gen_suffix_part = ""
    use_on_the_fly_texture = args.generate_texture_on_the_fly
    if not args.generate_texture_on_the_fly and not args.texture:
        print("Neither --texture file nor --generate_texture_on_the_fly specified. Defaulting to on-the-fly generation.")
        use_on_the_fly_texture = True

    if use_on_the_fly_texture:
        print("Preparing for on-the-fly texture generation...")
        if original_input_pil_for_texture_base: texture_base_image_pil = original_input_pil_for_texture_base.copy()
        else:
            try: texture_base_image_pil = Image.open(args.input).convert("RGB") 
            except Exception as e: print(f"Error loading base image for texture gen: {e}"); texture_base_image_pil = None
        
        if texture_base_image_pil:
            if args.tex_max_megapixels > 0:
                w_tex_base, h_tex_base = texture_base_image_pil.size
                if (w_tex_base * h_tex_base) / 1_000_000.0 > args.tex_max_megapixels:
                     texture_gen_suffix_part += f"_texResize{args.tex_max_megapixels:.1f}MP"
            if args.tex_method1_color_dots: texture_gen_suffix_part += "_texM1"
            if args.tex_method2_density_size: texture_gen_suffix_part += "_texM2"
            if args.tex_method3_voronoi: texture_gen_suffix_part += "_texM3"
            if args.tex_method4_glyph_dither: texture_gen_suffix_part += "_texM4"
            active_tex_methods = sum([args.tex_method1_color_dots, args.tex_method2_density_size, args.tex_method3_voronoi, args.tex_method4_glyph_dither])
            if active_tex_methods > 0:
                if active_tex_methods > 1 and args.tex_combination_mode == 'blend': texture_gen_suffix_part += f"_blend{args.tex_blend_type[0].upper()}"
                texture_to_use_pil = deeptexture.generate_texture_from_config(texture_base_image_pil, args, verbose=True)
            elif not texture_gen_suffix_part: 
                 print("No texture generation methods or resize selected for on-the-fly. Using input image as texture base directly.")
                 texture_to_use_pil = deeptexture.resize_to_megapixels(texture_base_image_pil, args.tex_max_megapixels) if args.tex_max_megapixels > 0 else texture_base_image_pil
            else: # Only tex_resize was active
                 texture_to_use_pil = deeptexture.resize_to_megapixels(texture_base_image_pil, args.tex_max_megapixels)

            if args.save_generated_texture and texture_to_use_pil:
                gen_tex_save_path = args.save_generated_texture
                if os.path.isdir(args.save_generated_texture) or not os.path.splitext(args.save_generated_texture)[1]:
                    gen_tex_filename = f"{output_filename_base}{filename_suffix}{texture_gen_suffix_part}_gentex_{timestamp}.png"
                    gen_tex_save_path = os.path.join(args.save_generated_texture if os.path.isdir(args.save_generated_texture) else args.output_dir, gen_tex_filename)
                try: 
                    os.makedirs(os.path.dirname(gen_tex_save_path), exist_ok=True)
                    texture_to_use_pil.save(gen_tex_save_path); print(f"Saved on-the-fly generated texture to {gen_tex_save_path}")
                except Exception as e: print(f"Error saving generated texture: {e}")
        else: print("Could not prepare base image for on-the-fly texture generation.")
    
    filename_suffix += texture_gen_suffix_part

    if texture_to_use_pil is None:
        if args.texture:
            try:
                print(f"Loading texture from file: {args.texture}")
                texture_to_use_pil = Image.open(args.texture).convert('RGB')
                filename_suffix += "_fileTex"
            except FileNotFoundError: print(f"Error: Texture file '{args.texture}' not found."); return
            except Exception as e: print(f"Error opening texture file: {e}"); return
        else:
            print("Warning: No texture source defined or successfully generated. Creating default random noise texture.")
            w_fb, h_fb = generated_depth_map_pil.size; noise_data = np.random.randint(0, 256, (h_fb, w_fb, 3), dtype=np.uint8)
            texture_to_use_pil = Image.fromarray(noise_data)
            filename_suffix += "_randomTex"

    if not texture_to_use_pil: print("Critical error: Texture could not be prepared. Exiting."); return

    final_stereogram_filename = f"{output_filename_base}{filename_suffix}_{timestamp}.png"
    final_stereogram_path = os.path.join(args.output_dir, final_stereogram_filename)

    print(f"Proceeding to stereogram generation. Output will be: {final_stereogram_path}")
    success = generate_stereogram_from_pil_texture(
        generated_depth_map_pil, texture_to_use_pil, final_stereogram_path, args.minsep, args.maxsep
    )

    if success: print(f"DeepStereo generation complete! Output: {final_stereogram_path}")
    else: print("DeepStereo generation failed.")

if __name__ == "__main__":
    main()
