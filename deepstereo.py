import argparse
from PIL import Image, ImageOps
import torch
import numpy as np
import cv2 # OpenCV for transformations
import os # For path operations
from datetime import datetime # For unique output names
from tqdm import tqdm 

import deeptexture # Import the refactored texture generation module

# Default stereogram separation values
MIN_SEPARATION_DEFAULT = 40 # Changed default
MAX_SEPARATION_DEFAULT = 100 # Changed default

# --- Stereogram Generation Function ---
# (generate_stereogram_from_pil_texture - no changes from last version)
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
# (create_depth_map_from_image - no changes from last version)
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
def main():
    parser = argparse.ArgumentParser(description="DeepStereo: AI-Powered Autostereogram Generator.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Main I/O
    parser.add_argument("--input", required=True, help="Path to the input color image (for depth and default texture base).")
    parser.add_argument("--output_dir", default="output_stereograms", help="Directory to save the generated stereogram image.")
    parser.add_argument("--output_filename_base", default=None, help="Optional base for output filename. If None, uses input filename base.")

    # Stereogram Params
    stereo_group = parser.add_argument_group('Stereogram Generation Parameters')
    stereo_group.add_argument("--minsep", type=int, default=MIN_SEPARATION_DEFAULT, help="Min separation for far points (pixels).") # Default updated
    stereo_group.add_argument("--maxsep", type=int, default=MAX_SEPARATION_DEFAULT, help="Max separation for near points (pixels).") # Default updated

    # Depth Map Params
    depth_group = parser.add_argument_group('Depth Map Generation (MiDaS)')
    depth_group.add_argument("--midasmodel", default="MiDaS_small", choices=["MiDaS_small", "DPT_Large", "DPT_Hybrid"], help="MiDaS model for depth estimation.")
    depth_group.add_argument("--save_depthmap", type=str, default=None, help="Optional: Path to save the AI-generated depth map (full path or directory).")
    depth_group.add_argument("--depth_model_input_width", type=int, default=384, help="Width to resize input for MiDaS (aspect preserved, rounded to 32px). 0 for no resize.")

    # Texture Source Params
    texture_source_group = parser.add_argument_group('Texture Source')
    texture_source_group.add_argument("--texture", default=None, help="Path to an external texture image. If None, on-the-fly generation is used.")
    texture_source_group.add_argument("--generate_texture_on_the_fly", action="store_true", help="Force on-the-fly texture generation. Overrides --texture if specified.")
    texture_source_group.add_argument("--texture_base_image_path", default=None, help="Optional path to a different image to use as the base for on-the-fly texture generation. If None, uses the main --input image.")
    texture_source_group.add_argument("--save_generated_texture", action="store_true", help="If true, saves the on-the-fly generated texture using default naming rules in --output_dir.")


    # On-the-fly Texture Generation Args (from deeptexture.py, prefixed with tex_)
    # These are the user overrides. If not provided, and on-the-fly is active, "preferred defaults" will be used.
    tex_gen_group = parser.add_argument_group('On-the-fly Texture Generation Overrides (used if --generate_texture_on_the_fly)')
    tex_gen_group.add_argument("--tex_max_megapixels", type=float, help="Texture: Resize base image for texture to approx this MP. (Default for auto-gen: 2.0)")
    tex_gen_group.add_argument("--tex_combination_mode", choices=["sequential", "blend"], help="Texture: How to combine method outputs. (Default for auto-gen: blend)")
    tex_gen_group.add_argument("--tex_blend_type", choices=["average", "lighten", "darken", "multiply", "screen", "add", "difference", "overlay"], help="Texture: Blend mode. (Default for auto-gen: average)")
    tex_gen_group.add_argument("--tex_blend_opacity", type=float, help="Texture: Blend opacity (0.0-1.0). (Default for auto-gen: 0.75)")
    
    tex_m1_group = tex_gen_group.add_argument_group('Texture Method 1 Overrides: Color Dots')
    tex_m1_group.add_argument("--tex_method1_color_dots", action="store_true", help="Texture M1: Apply. (Default for auto-gen: True)")
    tex_m1_group.add_argument("--tex_m1_density", type=float, help="Texture M1: Dot density. (Default: 0.7)")
    tex_m1_group.add_argument("--tex_m1_dot_size", type=int, help="Texture M1: Dot size. (Default for auto-gen: 50)")
    tex_m1_group.add_argument("--tex_m1_bg_color", type=str, help="Texture M1: BG color. (Default: black)")
    tex_m1_group.add_argument("--tex_m1_color_mode", choices=["content_pixel", "random_rgb", "random_from_palette", "transformed_hue", "transformed_invert"], help="Texture M1: Color mode. (Default for auto-gen: transformed_hue)")
    tex_m1_group.add_argument("--tex_m1_hue_shift_degrees", type=float, help="Texture M1: Hue shift. (Default: 90)")
    
    tex_m2_group = tex_gen_group.add_argument_group('Texture Method 2 Overrides: Density/Size Driven')
    tex_m2_group.add_argument("--tex_method2_density_size", action="store_true", help="Texture M2: Apply.")
    # ... (all other tex_m2, tex_m3, tex_m4 override args)
    tex_m2_group.add_argument("--tex_m2_mode", choices=["density", "size"], help="Texture M2: Mode. (Default: density)")
    tex_m2_group.add_argument("--tex_m2_element_color", type=str, help="Texture M2: Element color. (Default: white)")
    tex_m2_group.add_argument("--tex_m2_bg_color", type=str, help="Texture M2: BG color. (Default: black)")
    tex_m2_group.add_argument("--tex_m2_base_size", type=int, help="Texture M2: Base size. (Default: 3)")
    tex_m2_group.add_argument("--tex_m2_max_size", type=int, help="Texture M2: Max size. (Default: 12)")
    tex_m2_group.add_argument("--tex_m2_invert_influence", action="store_true", help="Texture M2: Invert influence.")
    tex_m2_group.add_argument("--tex_m2_density_factor", type=float, help="Texture M2: Density factor. (Default: 1.0)")

    tex_m3_group = tex_gen_group.add_argument_group('Texture Method 3 Overrides: Voronoi')
    tex_m3_group.add_argument("--tex_method3_voronoi", action="store_true", help="Texture M3: Apply.")
    tex_m3_group.add_argument("--tex_m3_num_points", type=int, help="Texture M3: Num points. (Default: 200)")
    tex_m3_group.add_argument("--tex_m3_metric", choices=["F1", "F2", "F2-F1"], help="Texture M3: Metric. (Default: F1)")
    tex_m3_group.add_argument("--tex_m3_color_source", choices=["distance", "content_point_color", "voronoi_cell_content_color"], help="Texture M3: Color source. (Default: distance)")

    tex_m4_group = tex_gen_group.add_argument_group('Texture Method 4 Overrides: Glyph Dither')
    tex_m4_group.add_argument("--tex_method4_glyph_dither", action="store_true", help="Texture M4: Apply. (Default for auto-gen: True)")
    tex_m4_group.add_argument("--tex_m4_num_colors", type=int, help="Texture M4: Num colors for quantization. (Default: 8)")
    tex_m4_group.add_argument("--tex_m4_glyph_size", type=int, help="Texture M4: Glyph size. (Default: 10)")
    tex_m4_group.add_argument("--tex_m4_glyph_style", choices=["random_dots", "lines", "circles", "solid"], help="Texture M4: Glyph style. (Default: random_dots)")
    tex_m4_group.add_argument("--tex_m4_use_quantized_color_for_glyph_element", action="store_true", help="Texture M4: Use quantized color for glyph elements.")


    args = parser.parse_args()
    print("--- DeepStereo Generator ---")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")

    if args.minsep >= args.maxsep: print("Error: --minsep must be less than --maxsep."); return

    input_filename_base = os.path.splitext(os.path.basename(args.input))[0]
    output_filename_base = args.output_filename_base if args.output_filename_base else input_filename_base
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_suffix = "" # This will be built based on operations

    # Determine target size for MiDaS
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

    # --- Texture Preparation ---
    texture_to_use_pil = None
    texture_base_image_source_path = args.texture_base_image_path if args.texture_base_image_path else args.input
    texture_gen_suffix_part = ""
    
    # Determine if on-the-fly texture generation is primary
    # If --generate_texture_on_the_fly is set, it takes precedence.
    # If --texture is not set, on-the-fly is implied.
    should_generate_texture = args.generate_texture_on_the_fly or (not args.texture)

    if should_generate_texture:
        print("Preparing for on-the-fly texture generation...")
        try:
            texture_base_image_pil = Image.open(texture_base_image_source_path).convert("RGB")
            print(f"Using texture base image: {texture_base_image_source_path}")
        except Exception as e:
            print(f"Error loading texture base image '{texture_base_image_source_path}': {e}")
            texture_base_image_pil = None # Fallback handled later

        if texture_base_image_pil:
            # Create a temporary args-like object for deeptexture, populating with user overrides or your preferred defaults
            tex_args_for_generator = argparse.Namespace()

            # Your preferred defaults if no specific tex_method flags are set by user
            preferred_defaults_active = not any([
                args.tex_method1_color_dots, args.tex_method2_density_size,
                args.tex_method3_voronoi, args.tex_method4_glyph_dither
            ])

            if preferred_defaults_active:
                print("Using preferred default settings for on-the-fly texture generation.")
                setattr(tex_args_for_generator, 'tex_max_megapixels', args.tex_max_megapixels if args.tex_max_megapixels is not None else 2.0)
                setattr(tex_args_for_generator, 'tex_combination_mode', args.tex_combination_mode if args.tex_combination_mode is not None else 'blend')
                setattr(tex_args_for_generator, 'tex_blend_type', args.tex_blend_type if args.tex_blend_type is not None else 'average')
                setattr(tex_args_for_generator, 'tex_blend_opacity', args.tex_blend_opacity if args.tex_blend_opacity is not None else 0.75)
                
                setattr(tex_args_for_generator, 'tex_method1_color_dots', True) # Default M1
                setattr(tex_args_for_generator, 'tex_m1_density', args.tex_m1_density if args.tex_m1_density is not None else 0.7)
                setattr(tex_args_for_generator, 'tex_m1_dot_size', args.tex_m1_dot_size if args.tex_m1_dot_size is not None else 50)
                setattr(tex_args_for_generator, 'tex_m1_bg_color', args.tex_m1_bg_color if args.tex_m1_bg_color is not None else "black")
                setattr(tex_args_for_generator, 'tex_m1_color_mode', args.tex_m1_color_mode if args.tex_m1_color_mode is not None else "transformed_hue")
                setattr(tex_args_for_generator, 'tex_m1_hue_shift_degrees', args.tex_m1_hue_shift_degrees if args.tex_m1_hue_shift_degrees is not None else 90)

                setattr(tex_args_for_generator, 'tex_method4_glyph_dither', True) # Default M4
                setattr(tex_args_for_generator, 'tex_m4_num_colors', args.tex_m4_num_colors if args.tex_m4_num_colors is not None else 8)
                setattr(tex_args_for_generator, 'tex_m4_glyph_size', args.tex_m4_glyph_size if args.tex_m4_glyph_size is not None else 10)
                setattr(tex_args_for_generator, 'tex_m4_glyph_style', args.tex_m4_glyph_style if args.tex_m4_glyph_style is not None else "random_dots")
                setattr(tex_args_for_generator, 'tex_m4_use_quantized_color_for_glyph_element', True if args.tex_m4_use_quantized_color_for_glyph_element else False) # Default for your pref
                
                # Ensure other methods are explicitly false if not overridden by user
                setattr(tex_args_for_generator, 'tex_method2_density_size', args.tex_method2_density_size)
                setattr(tex_args_for_generator, 'tex_method3_voronoi', args.tex_method3_voronoi)
            else: # User provided specific tex_method flags, use their settings or script defaults for those
                print("Using user-specified or script default settings for on-the-fly texture generation.")
                for arg_name, arg_value in vars(args).items():
                    if arg_name.startswith("tex_"):
                        setattr(tex_args_for_generator, arg_name, arg_value)
                # Ensure any non-specified tex_method booleans are false
                if not hasattr(tex_args_for_generator, 'tex_method1_color_dots'): setattr(tex_args_for_generator, 'tex_method1_color_dots', False)
                if not hasattr(tex_args_for_generator, 'tex_method2_density_size'): setattr(tex_args_for_generator, 'tex_method2_density_size', False)
                if not hasattr(tex_args_for_generator, 'tex_method3_voronoi'): setattr(tex_args_for_generator, 'tex_method3_voronoi', False)
                if not hasattr(tex_args_for_generator, 'tex_method4_glyph_dither'): setattr(tex_args_for_generator, 'tex_method4_glyph_dither', False)
                # Ensure essential general tex args have defaults if not provided by user when methods ARE specified
                if not hasattr(tex_args_for_generator, 'tex_max_megapixels') or tex_args_for_generator.tex_max_megapixels is None: setattr(tex_args_for_generator, 'tex_max_megapixels', 1.0)
                if not hasattr(tex_args_for_generator, 'tex_combination_mode')or tex_args_for_generator.tex_combination_mode is None: setattr(tex_args_for_generator, 'tex_combination_mode', 'sequential')
                if not hasattr(tex_args_for_generator, 'tex_blend_type')or tex_args_for_generator.tex_blend_type is None: setattr(tex_args_for_generator, 'tex_blend_type', 'overlay')
                if not hasattr(tex_args_for_generator, 'tex_blend_opacity') or tex_args_for_generator.tex_blend_opacity is None: setattr(tex_args_for_generator, 'tex_blend_opacity', 1.0)


            # Build texture_gen_suffix_part based on tex_args_for_generator
            if getattr(tex_args_for_generator, 'tex_max_megapixels', 0) > 0:
                w_tex_base, h_tex_base = texture_base_image_pil.size
                if (w_tex_base * h_tex_base) / 1_000_000.0 > tex_args_for_generator.tex_max_megapixels:
                     texture_gen_suffix_part += f"_texResize{tex_args_for_generator.tex_max_megapixels:.1f}MP"
            if getattr(tex_args_for_generator, 'tex_method1_color_dots', False): texture_gen_suffix_part += "_texM1"
            if getattr(tex_args_for_generator, 'tex_method2_density_size', False): texture_gen_suffix_part += "_texM2"
            if getattr(tex_args_for_generator, 'tex_method3_voronoi', False): texture_gen_suffix_part += "_texM3"
            if getattr(tex_args_for_generator, 'tex_method4_glyph_dither', False): texture_gen_suffix_part += "_texM4"
            
            active_tex_methods_count = sum([
                getattr(tex_args_for_generator, 'tex_method1_color_dots', False),
                getattr(tex_args_for_generator, 'tex_method2_density_size', False),
                getattr(tex_args_for_generator, 'tex_method3_voronoi', False),
                getattr(tex_args_for_generator, 'tex_method4_glyph_dither', False)
            ])

            if active_tex_methods_count > 0:
                if active_tex_methods_count > 1 and getattr(tex_args_for_generator, 'tex_combination_mode', 'sequential') == 'blend':
                    texture_gen_suffix_part += f"_blend{getattr(tex_args_for_generator, 'tex_blend_type', 'overlay')[0].upper()}"
                
                texture_to_use_pil = deeptexture.generate_texture_from_config(
                    texture_base_image_pil, tex_args_for_generator, verbose=True
                )
            elif not texture_gen_suffix_part: # No methods, no resize for texture
                 texture_to_use_pil = texture_base_image_pil 
            else: # Only tex_resize was active
                 texture_to_use_pil = deeptexture.resize_to_megapixels(texture_base_image_pil, tex_args_for_generator.tex_max_megapixels)
            
            if args.save_generated_texture and texture_to_use_pil:
                # Construct a filename for the generated texture similar to deeptexture standalone
                gen_tex_output_base = os.path.splitext(os.path.basename(texture_base_image_source_path))[0]
                gen_tex_filename = f"{gen_tex_output_base}{texture_gen_suffix_part}_gentex_{timestamp}.png"
                gen_tex_save_path = os.path.join(args.output_dir, gen_tex_filename) # Save in main output dir
                try: 
                    texture_to_use_pil.save(gen_tex_save_path); print(f"Saved on-the-fly generated texture to {gen_tex_save_path}")
                except Exception as e: print(f"Error saving generated texture: {e}")
        else:
            print("Could not prepare base image for on-the-fly texture generation.")
    
    filename_suffix += texture_gen_suffix_part

    if texture_to_use_pil is None: # If not generated, or generation failed
        if args.texture: # Fallback to --texture file
            try:
                print(f"Loading texture from file: {args.texture}")
                texture_to_use_pil = Image.open(args.texture).convert('RGB')
                filename_suffix += "_fileTex"
            except FileNotFoundError: print(f"Error: Texture file '{args.texture}' not found."); return
            except Exception as e: print(f"Error opening texture file: {e}"); return
        else: # Final fallback: default random noise
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
