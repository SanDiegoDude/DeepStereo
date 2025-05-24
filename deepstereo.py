import argparse
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageChops, ImageColor
import torch
import numpy as np
import cv2 # OpenCV for transformations
import os # For path operations
from datetime import datetime # For unique output names
from tqdm import tqdm 

import deeptexture # Import the refactored texture generation module

# Default stereogram separation values
MIN_SEPARATION_DEFAULT = 40 
MAX_SEPARATION_DEFAULT = 100 

verbose_main = True # Global for now, can be tied to a --verbose flag

# --- Texture Transform Functions ---
# (apply_texture_transforms - unchanged from previous version)
def apply_texture_transforms(image_pil, rotate_degrees=0, grid_rows=0, grid_cols=0, invert_colors=False):
    transformed_image = image_pil.copy()
    transform_suffix = ""
    if rotate_degrees != 0:
        if verbose_main: print(f"Texture Transform: Rotating texture by {rotate_degrees} degrees...")
        fillcolor = (0,0,0)
        if transformed_image.mode == 'RGBA': fillcolor = (0,0,0,0)
        transformed_image = transformed_image.rotate(rotate_degrees, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=fillcolor)
        transform_suffix += f"_rot{rotate_degrees}"
    if grid_rows > 0 and grid_cols > 0:
        if verbose_main: print(f"Texture Transform: Applying grid {grid_rows}x{grid_cols}...")
        original_width, original_height = transformed_image.size
        cell_width = original_width // grid_cols
        cell_height = original_height // grid_rows
        if cell_width > 0 and cell_height > 0:
            cell_texture = transformed_image.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
            grid_image = Image.new(transformed_image.mode, (original_width, original_height))
            for r in range(grid_rows):
                for c in range(grid_cols):
                    grid_image.paste(cell_texture, (c * cell_width, r * cell_height))
            transformed_image = grid_image
            transform_suffix += f"_grid{grid_rows}x{grid_cols}"
        else:
            if verbose_main: print("Warning: Grid dimensions result in zero-size cells. Skipping grid transform.")
    if invert_colors:
        if verbose_main: print("Texture Transform: Inverting texture colors...")
        if transformed_image.mode == 'L': transformed_image = ImageOps.invert(transformed_image)
        elif transformed_image.mode == 'RGB': transformed_image = ImageChops.invert(transformed_image)
        elif transformed_image.mode == 'RGBA':
            r,g,b,a = transformed_image.split(); r_inv, g_inv, b_inv = ImageChops.invert(r), ImageChops.invert(g), ImageChops.invert(b)
            transformed_image = Image.merge('RGBA', (r_inv, g_inv, b_inv, a))
        else: 
            try: transformed_image = ImageOps.invert(transformed_image)
            except Exception as e:
                if verbose_main: print(f"Warning: Could not directly invert colors for mode {transformed_image.mode}. Error: {e}")
        transform_suffix += "_invC"
    return transformed_image, transform_suffix


# --- Stereogram Generation Function (Original/Standard Algorithm) ---
def generate_stereogram_standard_texture(depth_map_pil, texture_pil, output_path, min_sep, max_sep):
    try:
        depth_map_img = depth_map_pil.convert('L')
        texture_img = texture_pil.convert('RGB')
    except Exception as e:
        print(f"Error preparing images for standard stereogram: {e}")
        return False
    width, height = depth_map_img.size
    texture_width, texture_height = texture_img.size
    stereogram_img = Image.new('RGB', (width, height))
    depth_pixels = depth_map_img.load()
    texture_pixels = texture_img.load()
    output_pixels = stereogram_img.load()

    for y in tqdm(range(height), desc="Stereogram Rows (Std Algo)", leave=False):
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
                output_pixels[x, y] = output_pixels[ref_x, y] # Smearing/pulling happens here
    try:
        stereogram_img.save(output_path)
        return True
    except Exception as e:
        print(f"Error saving output stereogram (standard): {e}")
        return False

# --- Stereogram Generation Function (Improved Texture Algorithm) ---
def generate_stereogram_improved_texture(depth_map_pil, texture_pil, output_path, min_sep, max_sep):
    try:
        depth_map_img = depth_map_pil.convert('L')
        wallpaper_img = texture_pil.convert('RGB')
    except Exception as e:
        print(f"Error preparing images for improved stereogram: {e}")
        return False

    width, height = depth_map_img.size
    wallpaper_width, wallpaper_height = wallpaper_img.size
    stereogram_img = Image.new('RGB', (width, height))
    depth_pixels = depth_map_img.load()
    wallpaper_pixels = wallpaper_img.load()
    output_pixels = stereogram_img.load()

    # True Magic Eye algorithm with continuous texture flow
    for y in tqdm(range(height), desc="Stereogram Rows (Alt Algo)", leave=False):
        # For each row, we'll build a constraint graph and solve it
        # to maintain maximum texture continuity
        
        # First, identify all the constraints
        constraints = []
        for x in range(width):
            depth_value_normalized = depth_pixels[x, y] / 255.0
            current_separation = int(min_sep + (max_sep - min_sep) * depth_value_normalized)
            current_separation = max(1, current_separation)
            
            if x >= current_separation:
                # x must match x - current_separation
                constraints.append((x - current_separation, x))
        
        # Now assign colors while maintaining continuity
        assigned = [False] * width
        
        # Start from the left and propagate rightward
        for x in range(width):
            if not assigned[x]:
                # Find all pixels that must have the same color as x
                same_color_group = set([x])
                to_process = [x]
                
                while to_process:
                    current = to_process.pop()
                    # Find all constraints involving current
                    for (a, b) in constraints:
                        if a == current and b not in same_color_group:
                            same_color_group.add(b)
                            to_process.append(b)
                        elif b == current and a not in same_color_group:
                            same_color_group.add(a)
                            to_process.append(a)
                
                # Assign color to the entire group
                # Use the leftmost pixel's position for texture lookup
                # This maintains continuity
                leftmost = min(same_color_group)
                wp_x = leftmost % wallpaper_width
                wp_y = y % wallpaper_height
                color = wallpaper_pixels[wp_x, wp_y]
                
                for px in same_color_group:
                    output_pixels[px, y] = color
                    assigned[px] = True
    
    try:
        stereogram_img.save(output_path)
        return True
    except Exception as e:
        print(f"Error saving output stereogram (improved): {e}")
        return False

# --- Depth Map Generation Function ---
# (create_depth_map_from_image - unchanged from previous version)
def create_depth_map_from_image(image_path, model_type="MiDaS_small", target_size=None, invert_depth_map=False, verbose=True):
    if verbose: print(f"Depth Map Gen: Loading MiDaS model ({model_type})...")
    try:
        model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid": transform = midas_transforms.dpt_transform
        elif model_type == "MiDaS_small": transform = midas_transforms.small_transform
        else: 
            if verbose: print(f"Warning: Unknown MiDaS model type '{model_type}'. Using small_transform.")
            transform = midas_transforms.small_transform
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
        processed_depth_normalized = 1.0 - depth_normalized 
        if invert_depth_map:
            if verbose: print("Depth Map Gen: Inverting depth map.")
            processed_depth_normalized = 1.0 - processed_depth_normalized
        depth_map_visual = (processed_depth_normalized * 255).astype(np.uint8)
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
    global verbose_main 
    parser = argparse.ArgumentParser(description="DeepStereo: AI-Powered Autostereogram Generator.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Main I/O
    parser.add_argument("--input", required=True, help="Path to the input color image (for depth and default texture base).")
    parser.add_argument("--output_dir", default="output_stereograms", help="Directory to save the generated stereogram image.")
    parser.add_argument("--output_filename_base", default=None, help="Optional base for output filename. If None, uses input filename base.")

    # Stereogram Params
    stereo_group = parser.add_argument_group('Stereogram Generation Parameters')
    stereo_group.add_argument("--minsep", type=int, default=MIN_SEPARATION_DEFAULT, help="Min separation for far points (pixels).")
    stereo_group.add_argument("--maxsep", type=int, default=MAX_SEPARATION_DEFAULT, help="Max separation for near points (pixels).")
    stereo_group.add_argument("--tex_alt_algo", action="store_true", help="Use alternative stereogram generation algorithm for improved texture continuity.")


    # Depth Map Params
    # ... (depth_group args unchanged from previous version) ...
    depth_group = parser.add_argument_group('Depth Map Generation (MiDaS)')
    depth_group.add_argument("--midasmodel", default="MiDaS_small", choices=["MiDaS_small", "DPT_Large", "DPT_Hybrid"], help="MiDaS model for depth estimation.")
    depth_group.add_argument("--save_depthmap", type=str, default=None, help="Optional: Path to save the AI-generated depth map (full path or directory).")
    depth_group.add_argument("--depth_model_input_width", type=int, default=None, help="Width to resize input for MiDaS (aspect preserved, rounded to 32px). Default: 384 for small, 0 (no resize) for Large/Hybrid.")
    depth_group.add_argument("--depth_invert", action="store_true", help="Invert the generated depth map (near becomes far and vice-versa).")

    # Texture Source Params
    # ... (texture_source_group args for loading/generating and final transforms unchanged from previous) ...
    texture_source_group = parser.add_argument_group('Texture Source & Final Transforms')
    texture_source_group.add_argument("--texture", default=None, help="Path to an external texture image. If None, on-the-fly generation is used.")
    texture_source_group.add_argument("--generate_texture_on_the_fly", action="store_true", help="Force on-the-fly texture generation. Overrides --texture if specified.")
    texture_source_group.add_argument("--texture_base_image_path", default=None, help="Optional path to a different image to use as the base for on-the-fly texture generation. If None, uses the main --input image.")
    texture_source_group.add_argument("--save_generated_texture", action="store_true", help="If true, saves the on-the-fly generated texture using default naming rules in --output_dir.")
    texture_source_group.add_argument("--tex_input_raw", action="store_true", help="For on-the-fly gen: use the (resized) texture base image directly, bypassing M1-M4 methods, before final transforms.")
    texture_source_group.add_argument("--tex_rotate", type=int, default=0, help="Rotate final texture by DEGREES (0-359). Applied after generation/loading.")
    texture_source_group.add_argument("--tex_grid", type=str, default="0,0", help="Create a ROWS,COLS grid from the final texture. E.g., '2,2'. Applied after rotate.")
    texture_source_group.add_argument("--tex_invert_colors", action="store_true", help="Invert colors of the final texture. Applied after grid.")

    # On-the-fly Texture Generation Args
    # ... (tex_gen_group and its subgroups for M1-M4 args unchanged from previous) ...
    tex_gen_group = parser.add_argument_group('On-the-fly Texture Generation Overrides (used if --generate_texture_on_the_fly and --tex_input_raw is False)')
    tex_gen_group.add_argument("--tex_max_megapixels", type=float, default=None, help="Texture: Resize base image for texture to approx this MP. (Default for auto-gen: 2.0, for manual override: 1.0)")
    tex_gen_group.add_argument("--tex_combination_mode", type=str, choices=["sequential", "blend"], default=None, help="Texture: How to combine method outputs. (Default for auto-gen: blend, for manual override: sequential)")
    tex_gen_group.add_argument("--tex_blend_type", type=str, choices=["average", "lighten", "darken", "multiply", "screen", "add", "difference", "overlay"], default=None, help="Texture: Blend mode. (Default for auto-gen: average, for manual override: overlay)")
    tex_gen_group.add_argument("--tex_blend_opacity", type=float, default=None, help="Texture: Blend opacity (0.0-1.0). (Default for auto-gen: 0.75, for manual override: 1.0)")
    tex_m1_group = tex_gen_group.add_argument_group('Texture Method 1 Overrides: Color Dots')
    tex_m1_group.add_argument("--tex_method1_color_dots", action="store_true", help="Texture M1: Apply. (Default for auto-gen: True)")
    tex_m1_group.add_argument("--tex_m1_density", type=float, default=None, help="Texture M1: Dot density. (Script Default: 0.7)")
    tex_m1_group.add_argument("--tex_m1_dot_size", type=int, default=None, help="Texture M1: Dot size. (Default for auto-gen: 50, Script Default: 2)")
    tex_m1_group.add_argument("--tex_m1_bg_color", type=str, default=None, help="Texture M1: BG color. (Script Default: black)")
    tex_m1_group.add_argument("--tex_m1_color_mode", type=str, choices=["content_pixel", "random_rgb", "random_from_palette", "transformed_hue", "transformed_invert"], default=None, help="Texture M1: Color mode. (Default for auto-gen: transformed_hue, Script Default: content_pixel)")
    tex_m1_group.add_argument("--tex_m1_hue_shift_degrees", type=float, default=None, help="Texture M1: Hue shift. (Script Default: 90)")
    tex_m2_group = tex_gen_group.add_argument_group('Texture Method 2 Overrides: Density/Size Driven')
    tex_m2_group.add_argument("--tex_method2_density_size", action="store_true", help="Texture M2: Apply.")
    tex_m2_group.add_argument("--tex_m2_mode", type=str, choices=["density", "size"], default=None, help="Texture M2: Mode. (Script Default: density)")
    tex_m2_group.add_argument("--tex_m2_element_color", type=str, default=None, help="Texture M2: Element color. (Script Default: white)")
    tex_m2_group.add_argument("--tex_m2_bg_color", type=str, default=None, help="Texture M2: BG color. (Script Default: black)")
    tex_m2_group.add_argument("--tex_m2_base_size", type=int, default=None, help="Texture M2: Base size. (Script Default: 3)")
    tex_m2_group.add_argument("--tex_m2_max_size", type=int, default=None, help="Texture M2: Max size. (Script Default: 12)")
    tex_m2_group.add_argument("--tex_m2_invert_influence", action="store_true", help="Texture M2: Invert influence.") 
    tex_m2_group.add_argument("--tex_m2_density_factor", type=float, default=None, help="Texture M2: Density factor. (Script Default: 1.0)")
    tex_m3_group = tex_gen_group.add_argument_group('Texture Method 3 Overrides: Voronoi')
    tex_m3_group.add_argument("--tex_method3_voronoi", action="store_true", help="Texture M3: Apply.")
    tex_m3_group.add_argument("--tex_m3_num_points", type=int, default=None, help="Texture M3: Num points. (Script Default: 200)")
    tex_m3_group.add_argument("--tex_m3_metric", type=str, choices=["F1", "F2", "F2-F1"], default=None, help="Texture M3: Metric. (Script Default: F1)")
    tex_m3_group.add_argument("--tex_m3_color_source", type=str, choices=["distance", "content_point_color", "voronoi_cell_content_color"], default=None, help="Texture M3: Color source. (Script Default: distance)")
    tex_m4_group = tex_gen_group.add_argument_group('Texture Method 4 Overrides: Glyph Dither')
    tex_m4_group.add_argument("--tex_method4_glyph_dither", action="store_true", help="Texture M4: Apply. (Default for auto-gen: True)")
    tex_m4_group.add_argument("--tex_m4_num_colors", type=int, default=None, help="Texture M4: Num colors for quantization. (Script Default: 8)")
    tex_m4_group.add_argument("--tex_m4_glyph_size", type=int, default=None, help="Texture M4: Glyph size. (Script Default: 10)")
    tex_m4_group.add_argument("--tex_m4_glyph_style", type=str, choices=["random_dots", "lines", "circles", "solid"], default=None, help="Texture M4: Glyph style. (Script Default: random_dots)")
    tex_m4_group.add_argument("--tex_m4_use_quantized_color_for_glyph_element", action="store_true", help="Texture M4: Use quantized color. (Default for auto-gen: True)")


    args = parser.parse_args()
    print("--- DeepStereo Generator ---")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")

    if args.minsep >= args.maxsep: print("Error: --minsep must be less than --maxsep."); return

    input_filename_base = os.path.splitext(os.path.basename(args.input))[0]
    output_filename_base = args.output_filename_base if args.output_filename_base else input_filename_base
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_suffix = "" 

    depth_model_input_width_to_use = args.depth_model_input_width
    if args.depth_model_input_width is None: 
        if args.midasmodel in ["DPT_Large", "DPT_Hybrid"]:
            depth_model_input_width_to_use = 0 
            print("Using full resolution for DPT_Large/DPT_Hybrid model (depth_model_input_width=0).")
        else:
            depth_model_input_width_to_use = 384 
    
    target_midas_processing_size = None
    if depth_model_input_width_to_use and depth_model_input_width_to_use > 0: 
        try:
            with Image.open(args.input) as temp_img: orig_w, orig_h = temp_img.size
            aspect_ratio = orig_h / orig_w
            target_w_midas = (depth_model_input_width_to_use // 32) * 32
            target_h_midas = (int(target_w_midas * aspect_ratio) // 32) * 32
            if target_w_midas > 0 and target_h_midas > 0: target_midas_processing_size = (target_w_midas, target_h_midas)
            else: print(f"Warning: Calculated MiDaS processing size invalid. Using original size for depth map.")
        except Exception as e: print(f"Warning: Could not get input image size for MiDaS resize. Error: {e}.")
    
    generated_depth_map_pil, original_input_pil_for_texture_base = create_depth_map_from_image(
        args.input, model_type=args.midasmodel, target_size=target_midas_processing_size, invert_depth_map=args.depth_invert
    )

    if not generated_depth_map_pil: print("Could not generate depth map. Exiting."); return
    filename_suffix += f"_{args.midasmodel}"
    if target_midas_processing_size: filename_suffix += f"_depth{depth_model_input_width_to_use}w" # Use the effective width
    if args.depth_invert: filename_suffix += "_depthInv"

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
    texture_base_image_source_path = args.texture_base_image_path if args.texture_base_image_path else args.input
    texture_gen_suffix_part = "" # Suffix for texture generation steps
    final_transform_suffix_part = "" # Suffix for final texture transforms (rotate, grid, invert)
    
    should_generate_texture_methods = args.generate_texture_on_the_fly or (not args.texture)

    if should_generate_texture_methods:
        print("Preparing for on-the-fly texture processing...")
        try:
            texture_base_image_pil = Image.open(texture_base_image_source_path).convert("RGB")
            print(f"Using texture base image: {texture_base_image_source_path}")
        except Exception as e:
            print(f"Error loading texture base image '{texture_base_image_source_path}': {e}")
            texture_base_image_pil = None 

        if texture_base_image_pil:
            if args.tex_input_raw:
                print("Using raw texture base image (M1-M4 methods bypassed).")
                raw_tex_max_mp = args.tex_max_megapixels if args.tex_max_megapixels is not None else 1.0 
                texture_to_use_pil = deeptexture.resize_to_megapixels(texture_base_image_pil.copy(), raw_tex_max_mp, verbose=verbose_main)
                if raw_tex_max_mp > 0 and texture_to_use_pil.size != texture_base_image_pil.size :
                    texture_gen_suffix_part += f"_texRawResize{raw_tex_max_mp:.1f}MP"
                else:
                    texture_gen_suffix_part += "_texRaw"
            else: # Apply M1-M4 methods
                tex_args_for_generator = argparse.Namespace()
                user_set_any_tex_method_flag = any([
                    args.tex_method1_color_dots, args.tex_method2_density_size,
                    args.tex_method3_voronoi, args.tex_method4_glyph_dither
                ])
                use_preferred_defaults = not user_set_any_tex_method_flag

                if use_preferred_defaults:
                    print("Using preferred default settings for on-the-fly texture generation methods.")
                    # ... (Set preferred defaults on tex_args_for_generator as before) ...
                    setattr(tex_args_for_generator, 'tex_max_megapixels', 2.0)
                    setattr(tex_args_for_generator, 'tex_combination_mode', 'blend')
                    setattr(tex_args_for_generator, 'tex_blend_type', 'average')
                    setattr(tex_args_for_generator, 'tex_blend_opacity', 0.75)
                    setattr(tex_args_for_generator, 'tex_method1_color_dots', True)
                    setattr(tex_args_for_generator, 'tex_m1_density', 0.7)
                    setattr(tex_args_for_generator, 'tex_m1_dot_size', 50)
                    setattr(tex_args_for_generator, 'tex_m1_bg_color', "black")
                    setattr(tex_args_for_generator, 'tex_m1_color_mode', "transformed_hue")
                    setattr(tex_args_for_generator, 'tex_m1_hue_shift_degrees', 90.0)
                    setattr(tex_args_for_generator, 'tex_method4_glyph_dither', True)
                    setattr(tex_args_for_generator, 'tex_m4_num_colors', 8)
                    setattr(tex_args_for_generator, 'tex_m4_glyph_size', 10)
                    setattr(tex_args_for_generator, 'tex_m4_glyph_style', "random_dots")
                    setattr(tex_args_for_generator, 'tex_m4_use_quantized_color_for_glyph_element', True)
                    setattr(tex_args_for_generator, 'tex_method2_density_size', False) 
                    setattr(tex_args_for_generator, 'tex_method3_voronoi', False)
                else: 
                    print("Using user-specified flags for on-the-fly texture generation methods.")
                    # ... (Populate tex_args_for_generator from args or script defaults as before) ...
                    def get_arg_val(arg_short_name, script_default):
                        user_val = getattr(args, f"tex_{arg_short_name}", None)
                        return user_val if user_val is not None else script_default
                    setattr(tex_args_for_generator, 'tex_max_megapixels', get_arg_val('max_megapixels', 1.0))
                    setattr(tex_args_for_generator, 'tex_combination_mode', get_arg_val('combination_mode', 'sequential'))
                    setattr(tex_args_for_generator, 'tex_blend_type', get_arg_val('blend_type', 'overlay'))
                    setattr(tex_args_for_generator, 'tex_blend_opacity', get_arg_val('blend_opacity', 1.0))
                    setattr(tex_args_for_generator, 'tex_method1_color_dots', args.tex_method1_color_dots)
                    if args.tex_method1_color_dots:
                        setattr(tex_args_for_generator, 'tex_m1_density', get_arg_val('m1_density', 0.7))
                        setattr(tex_args_for_generator, 'tex_m1_dot_size', get_arg_val('m1_dot_size', 2))
                        setattr(tex_args_for_generator, 'tex_m1_bg_color', get_arg_val('m1_bg_color', "black"))
                        setattr(tex_args_for_generator, 'tex_m1_color_mode', get_arg_val('m1_color_mode', "content_pixel"))
                        setattr(tex_args_for_generator, 'tex_m1_hue_shift_degrees', get_arg_val('m1_hue_shift_degrees', 90.0))
                    setattr(tex_args_for_generator, 'tex_method2_density_size', args.tex_method2_density_size)
                    if args.tex_method2_density_size:
                        setattr(tex_args_for_generator, 'tex_m2_mode', get_arg_val('m2_mode', "density"))
                        setattr(tex_args_for_generator, 'tex_m2_element_color', get_arg_val('m2_element_color', "white"))
                        setattr(tex_args_for_generator, 'tex_m2_bg_color', get_arg_val('m2_bg_color', "black"))
                        setattr(tex_args_for_generator, 'tex_m2_base_size', get_arg_val('m2_base_size', 3))
                        setattr(tex_args_for_generator, 'tex_m2_max_size', get_arg_val('m2_max_size', 12))
                        setattr(tex_args_for_generator, 'tex_m2_invert_influence', args.tex_m2_invert_influence)
                        setattr(tex_args_for_generator, 'tex_m2_density_factor', get_arg_val('m2_density_factor', 1.0))
                    setattr(tex_args_for_generator, 'tex_method3_voronoi', args.tex_method3_voronoi)
                    if args.tex_method3_voronoi:
                        setattr(tex_args_for_generator, 'tex_m3_num_points', get_arg_val('m3_num_points', 200))
                        setattr(tex_args_for_generator, 'tex_m3_metric', get_arg_val('m3_metric', "F1"))
                        setattr(tex_args_for_generator, 'tex_m3_color_source', get_arg_val('m3_color_source', "distance"))
                    setattr(tex_args_for_generator, 'tex_method4_glyph_dither', args.tex_method4_glyph_dither)
                    if args.tex_method4_glyph_dither:
                        setattr(tex_args_for_generator, 'tex_m4_num_colors', get_arg_val('m4_num_colors', 8))
                        setattr(tex_args_for_generator, 'tex_m4_glyph_size', get_arg_val('m4_glyph_size', 10))
                        setattr(tex_args_for_generator, 'tex_m4_glyph_style', get_arg_val('m4_glyph_style', "random_dots"))
                        setattr(tex_args_for_generator, 'tex_m4_use_quantized_color_for_glyph_element', args.tex_m4_use_quantized_color_for_glyph_element)

                # Build suffix part for M1-M4 methods
                current_tex_max_mp_methods = getattr(tex_args_for_generator, 'tex_max_megapixels', 0)
                if current_tex_max_mp_methods > 0:
                    w_tex_base, h_tex_base = texture_base_image_pil.size
                    if (w_tex_base * h_tex_base) / 1_000_000.0 > current_tex_max_mp_methods:
                         texture_gen_suffix_part += f"_texResize{current_tex_max_mp_methods:.1f}MP"
                if getattr(tex_args_for_generator, 'tex_method1_color_dots', False): texture_gen_suffix_part += "_texM1"
                if getattr(tex_args_for_generator, 'tex_method2_density_size', False): texture_gen_suffix_part += "_texM2"
                if getattr(tex_args_for_generator, 'tex_method3_voronoi', False): texture_gen_suffix_part += "_texM3"
                if getattr(tex_args_for_generator, 'tex_method4_glyph_dither', False): texture_gen_suffix_part += "_texM4"
                active_tex_methods_count = sum([getattr(tex_args_for_generator, flag, False) for flag in ['tex_method1_color_dots', 'tex_method2_density_size', 'tex_method3_voronoi', 'tex_method4_glyph_dither']])
                if active_tex_methods_count > 0:
                    if active_tex_methods_count > 1 and getattr(tex_args_for_generator, 'tex_combination_mode', 'sequential') == 'blend':
                        texture_gen_suffix_part += f"_blend{getattr(tex_args_for_generator, 'tex_blend_type', 'overlay')[0].upper()}"
                    texture_to_use_pil = deeptexture.generate_texture_from_config(
                        texture_base_image_pil, tex_args_for_generator, verbose=verbose_main
                    )
                elif not texture_gen_suffix_part : texture_to_use_pil = texture_base_image_pil 
                else: texture_to_use_pil = deeptexture.resize_to_megapixels(texture_base_image_pil, current_tex_max_mp_methods, verbose=verbose_main)
            
            # Apply final transforms (rotate, grid, invert) AFTER M1-M4 or raw processing
            if texture_to_use_pil:
                try:
                    grid_r_str, grid_c_str = args.tex_grid.split(',')
                    grid_r, grid_c = int(grid_r_str), int(grid_c_str)
                except ValueError:
                    if verbose_main: print(f"Warning: Invalid format for --tex_grid '{args.tex_grid}'. Disabling grid."); grid_r, grid_c = 0,0
                texture_to_use_pil, final_transform_suffix_part = apply_texture_transforms(
                    texture_to_use_pil, args.tex_rotate, grid_r, grid_c, args.tex_invert_colors
                )
            else: 
                print("Error: Base texture for on-the-fly processing is missing, cannot apply final transforms.")

            if args.save_generated_texture and texture_to_use_pil:
                gen_tex_output_base = os.path.splitext(os.path.basename(texture_base_image_source_path))[0]
                # Suffix includes M1-M4 part and final transform part
                full_texture_suffix = texture_gen_suffix_part + final_transform_suffix_part
                gen_tex_filename = f"{gen_tex_output_base}{full_texture_suffix}_gentex_{timestamp}.png"
                gen_tex_save_path = os.path.join(args.output_dir, gen_tex_filename) 
                try: 
                    texture_to_use_pil.save(gen_tex_save_path); print(f"Saved on-the-fly generated texture to {gen_tex_save_path}")
                except Exception as e: print(f"Error saving generated texture: {e}")
        else:
            print("Could not prepare base image for on-the-fly texture generation.")
    
    filename_suffix += texture_gen_suffix_part + final_transform_suffix_part # Add all texture processing suffixes

    if texture_to_use_pil is None: 
        if args.texture: 
            try:
                print(f"Loading texture from file: {args.texture}")
                texture_to_use_pil = Image.open(args.texture).convert('RGB')
                # Apply final transforms to loaded file texture too
                grid_r_str, grid_c_str = args.tex_grid.split(',')
                grid_r, grid_c = int(grid_r_str), int(grid_c_str)
                texture_to_use_pil, file_tex_transform_suffix = apply_texture_transforms(
                    texture_to_use_pil, args.tex_rotate, grid_r, grid_c, args.tex_invert_colors
                )
                filename_suffix += "_fileTex" + file_tex_transform_suffix
            except FileNotFoundError: print(f"Error: Texture file '{args.texture}' not found."); return
            except Exception as e: print(f"Error opening or transforming texture file: {e}"); return
        else: 
            print("Warning: No texture source. Creating default random noise texture.")
            w_fb, h_fb = generated_depth_map_pil.size; noise_data = np.random.randint(0, 256, (h_fb, w_fb, 3), dtype=np.uint8)
            texture_to_use_pil = Image.fromarray(noise_data)
            filename_suffix += "_randomTex"

    if not texture_to_use_pil: print("Critical error: Texture could not be prepared. Exiting."); return

    # Add stereogram algorithm type to suffix
    if args.tex_alt_algo:
        filename_suffix += "_altAlgo"
    else:
        filename_suffix += "_stdAlgo"


    final_stereogram_filename = f"{output_filename_base}{filename_suffix}_{timestamp}.png"
    final_stereogram_path = os.path.join(args.output_dir, final_stereogram_filename)

    print(f"Proceeding to stereogram generation. Output will be: {final_stereogram_path}")
    
    stereogram_function_to_call = generate_stereogram_improved_texture if args.tex_alt_algo else generate_stereogram_standard_texture
    
    success = stereogram_function_to_call(
        generated_depth_map_pil, texture_to_use_pil, final_stereogram_path, args.minsep, args.maxsep
    )

    if success: print(f"DeepStereo generation complete! Output: {final_stereogram_path}")
    else: print("DeepStereo generation failed.")

if __name__ == "__main__":
    main()
