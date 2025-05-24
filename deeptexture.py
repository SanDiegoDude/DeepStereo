import argparse
import os
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import numpy as np
import random
import math
from tqdm import tqdm # Import tqdm

# --- Helper Functions ---
def parse_color(color_str, default_color=(0, 0, 0)):
    if ',' in color_str:
        try:
            return tuple(map(int, color_str.split(',')))
        except ValueError:
            return default_color
    elif color_str.lower() == "white":
        return (255, 255, 255)
    elif color_str.lower() == "black":
        return (0, 0, 0)
    elif color_str.lower() == "red":
        return (255, 0, 0)
    elif color_str.lower() == "green":
        return (0, 255, 0)
    elif color_str.lower() == "blue":
        return (0, 0, 255)
    return default_color

def get_pixel_value_safe(pil_image, x, y):
    x = max(0, min(x, pil_image.width - 1))
    y = max(0, min(y, pil_image.height - 1))
    return pil_image.getpixel((x, y))

# --- Method 1: Content-Image-Driven Color Palette for Procedural Dots ---
def apply_method1_color_dots(content_image_pil, density=0.5, dot_size=1, bg_color=(0,0,0)):
    print(f"Applying Method 1: Color Dots (Density: {density}, Size: {dot_size})")
    width, height = content_image_pil.size
    output_image = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(output_image)
    num_dots = int(width * height * density)

    # Wrap the loop with tqdm
    for _ in tqdm(range(num_dots), desc="Method 1: Dots"):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        content_color = get_pixel_value_safe(content_image_pil.convert("RGB"), x, y)
        if dot_size == 1:
            draw.point((x, y), fill=content_color)
        else:
            draw.rectangle([x, y, x + dot_size - 1, y + dot_size - 1], fill=content_color)
    return output_image

# --- Method 2: Content-Image-Driven Density/Size of Procedural Elements ---
def apply_method2_density_size(content_image_pil, mode="density", element_color=(0,0,0),
                               bg_color=(255,255,255), base_size=2, max_size=10,
                               invert_influence=False, density_factor=1.0):
    print(f"Applying Method 2: {mode.capitalize()} (Element: {element_color}, BG: {bg_color}, Invert: {invert_influence})")
    width, height = content_image_pil.size
    output_image = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(output_image)
    content_gray = content_image_pil.convert("L")

    if mode == "density":
        step = max(1, base_size)
        # Wrap the outer loop with tqdm
        for r in tqdm(range(0, height, step), desc="Method 2: Density"):
            for c in range(0, width, step):
                brightness = get_pixel_value_safe(content_gray, c, r) / 255.0
                influence = brightness if invert_influence else 1.0 - brightness
                if random.random() < (influence * density_factor):
                    draw.rectangle([c, r, c + base_size - 1, r + base_size - 1], fill=element_color)
    elif mode == "size":
        num_elements = int((width * height * 0.1) * density_factor)
        # Wrap the loop with tqdm
        for _ in tqdm(range(num_elements), desc="Method 2: Size"):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            brightness = get_pixel_value_safe(content_gray, x, y) / 255.0
            influence = brightness if invert_influence else 1.0 - brightness
            current_size = max(1, int(base_size + (max_size - base_size) * influence))
            ex1, ey1 = x - current_size // 2, y - current_size // 2
            ex2, ey2 = ex1 + current_size - 1, ey1 + current_size - 1
            draw.rectangle([ex1, ey1, ex2, ey2], fill=element_color)
    return output_image

# --- Method 3: Simplified Voronoi/Worley-like Noise ---
def apply_method3_voronoi(content_image_pil, num_points=100, metric="F1",
                          color_source="distance", point_color=(255,0,0)):
    print(f"Applying Method 3: Voronoi-like (Points: {num_points}, Metric: {metric}, Color: {color_source})")
    width, height = content_image_pil.size
    output_image = Image.new("RGB", (width, height))
    output_pixels = output_image.load()
    points = []
    for _ in range(num_points):
        px, py = random.randint(0, width - 1), random.randint(0, height - 1)
        p_content_color = get_pixel_value_safe(content_image_pil.convert("RGB"), px, py)
        points.append((px, py, p_content_color))

    if not points: return content_image_pil.copy()
    max_dist_val = math.sqrt(width**2 + height**2) / 3 # Adjusted normalization factor

    # Wrap the outer loop with tqdm
    for r in tqdm(range(height), desc="Method 3: Voronoi"):
        for c in range(width):
            distances_sq = sorted([((c - px)**2 + (r - py)**2, p_content_color) for px, py, p_content_color in points])
            final_color = (0,0,0)
            if color_source == "distance":
                dist_val = 0
                if metric == "F1" and len(distances_sq) > 0: dist_val = math.sqrt(distances_sq[0][0])
                elif metric == "F2" and len(distances_sq) > 1: dist_val = math.sqrt(distances_sq[1][0])
                elif metric == "F2-F1" and len(distances_sq) > 1: dist_val = abs(math.sqrt(distances_sq[1][0]) - math.sqrt(distances_sq[0][0]))
                norm_dist = min(dist_val / max_dist_val, 1.0)
                gray_val = int(norm_dist * 255)
                final_color = (gray_val, gray_val, gray_val)
            elif color_source == "content_point_color":
                idx = 0
                if metric == "F2" and len(distances_sq) > 1: idx = 1
                # F2-F1 for color is less intuitive, here just using F1/F2's point color
                final_color = distances_sq[idx][1] if len(distances_sq) > idx else (0,0,0)
            elif color_source == "content_pixel_color": # Actually Voronoi cell coloring
                final_color = distances_sq[0][1] if distances_sq else (0,0,0)
            output_pixels[c, r] = final_color
    return output_image

# --- Method 4: Stylized Dithering with Custom Glyphs ---
def generate_glyph(glyph_style, size, color=(0,0,0), bg_color=(255,255,255)):
    glyph_img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(glyph_img)
    if glyph_style == "random_dots":
        for _ in range(int(size * size * 0.3)):
            draw.point((random.randint(0, size - 1), random.randint(0, size - 1)), fill=color)
    elif glyph_style == "lines":
        for i in range(0, size, max(1, size // 4)): draw.line([(0, i), (size - 1, i)], fill=color, width=1)
    elif glyph_style == "circles":
        draw.ellipse([(1, 1, size - 2, size - 2)], outline=color, fill=bg_color if random.random() > 0.5 else color)
    elif glyph_style == "solid":
        draw.rectangle([(0,0), (size-1,size-1)], fill=color)
    return glyph_img

def apply_method4_glyph_dither(content_image_pil, num_colors=4, glyph_size=8,
                               glyph_style="random_dots", base_glyph_color_str="0,0,0",
                               use_quantized_color_for_glyph=True):
    print(f"Applying Method 4: Glyph Dither (Colors: {num_colors}, Size: {glyph_size}, Style: {glyph_style})")
    width, height = content_image_pil.size
    quantized_content = content_image_pil.convert("RGB").quantize(colors=num_colors, method=Image.Quantize.MAXCOVERAGE)
    quantized_content_rgb = quantized_content.convert("RGB")
    output_image = Image.new("RGB", (width, height))
    base_glyph_color = parse_color(base_glyph_color_str)

    # Wrap the outer loop with tqdm
    for r_block in tqdm(range(0, height, glyph_size), desc="Method 4: Glyph Dither"):
        for c_block in range(0, width, glyph_size):
            block_x = min(c_block + glyph_size // 2, width - 1)
            block_y = min(r_block + glyph_size // 2, height - 1)
            quantized_pixel_color = get_pixel_value_safe(quantized_content_rgb, block_x, block_y)
            glyph_draw_color = quantized_pixel_color if use_quantized_color_for_glyph else base_glyph_color
            glyph_bg_color = (0,0,0) if sum(glyph_draw_color) > 384 else (255,255,255)
            glyph = generate_glyph(glyph_style, glyph_size, color=glyph_draw_color, bg_color=glyph_bg_color)
            output_image.paste(glyph, (c_block, r_block))
    return output_image

# --- Main Execution ---
def main():
    print("--- Procedural Texture Generator ---")
    print("NOTE: All processing is done on the CPU and can be slow for large images or complex methods.")
    parser = argparse.ArgumentParser(description="Procedurally generate textures influenced by a content image.")
    parser.add_argument("--input", required=True, help="Path to the input content image.")
    parser.add_argument("--output_dir", default=".", help="Directory to save the processed image.")
    
    # Method 1 Args
    parser.add_argument("--method1_color_dots", action="store_true", help="Apply Method 1: Content-driven color dots.")
    parser.add_argument("--m1_density", type=float, default=0.6, help="Dot density for Method 1 (0.0 to 1.0).")
    parser.add_argument("--m1_dot_size", type=int, default=1, help="Dot size for Method 1 (pixels).")
    parser.add_argument("--m1_bg_color", type=str, default="0,0,0", help="Background color for Method 1 (R,G,B or name).")

    # Method 2 Args
    parser.add_argument("--method2_density_size", action="store_true", help="Apply Method 2: Content-driven density or size of elements.")
    parser.add_argument("--m2_mode", choices=["density", "size"], default="density", help="Mode for Method 2.")
    parser.add_argument("--m2_element_color", type=str, default="255,255,255", help="Element color for Method 2 (R,G,B or name).")
    parser.add_argument("--m2_bg_color", type=str, default="0,0,0", help="Background color for Method 2 (R,G,B or name).")
    parser.add_argument("--m2_base_size", type=int, default=2, help="Base size for elements in Method 2.")
    parser.add_argument("--m2_max_size", type=int, default=10, help="Max size for elements in Method 2 (size mode).")
    parser.add_argument("--m2_invert_influence", action="store_true", help="Invert content brightness influence for Method 2.")
    parser.add_argument("--m2_density_factor", type=float, default=1.0, help="Overall density factor for Method 2.")

    # Method 3 Args
    parser.add_argument("--method3_voronoi", action="store_true", help="Apply Method 3: Simplified Voronoi/Worley-like noise.")
    parser.add_argument("--m3_num_points", type=int, default=150, help="Number of seed points for Method 3.")
    parser.add_argument("--m3_metric", choices=["F1", "F2", "F2-F1"], default="F1", help="Distance metric for Method 3 (F1=closest, F2=2nd, F2-F1).")
    parser.add_argument("--m3_color_source", choices=["distance", "content_point_color", "content_pixel_color"], default="distance", help="Color source for Method 3 cells.")
    
    # Method 4 Args
    parser.add_argument("--method4_glyph_dither", action="store_true", help="Apply Method 4: Stylized dithering with glyphs.")
    parser.add_argument("--m4_num_colors", type=int, default=4, help="Number of colors for quantization in Method 4.")
    parser.add_argument("--m4_glyph_size", type=int, default=8, help="Size of glyphs for Method 4 (pixels).")
    parser.add_argument("--m4_glyph_style", choices=["random_dots", "lines", "circles", "solid"], default="random_dots", help="Style of glyphs for Method 4.")
    parser.add_argument("--m4_base_glyph_color", type=str, default="0,0,0", help="Base color for glyph elements if not using quantized color.")
    parser.add_argument("--m4_use_quantized_color", action="store_true", help="Use quantized color from content image for glyph elements.")


    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input image '{args.input}' not found.")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(args.input))[0]
    
    try:
        current_image = Image.open(args.input)
        print(f"Loaded input image: {args.input} (Size: {current_image.size})")
    except Exception as e:
        print(f"Error loading input image: {e}")
        return

    processed_suffix = ""

    if args.method1_color_dots:
        current_image = apply_method1_color_dots(current_image, args.m1_density, args.m1_dot_size, parse_color(args.m1_bg_color))
        processed_suffix += "_m1"
    
    if args.method2_density_size:
        current_image = apply_method2_density_size(current_image, args.m2_mode, parse_color(args.m2_element_color),
                                                 parse_color(args.m2_bg_color), args.m2_base_size, args.m2_max_size,
                                                 args.m2_invert_influence, args.m2_density_factor)
        processed_suffix += "_m2"

    if args.method3_voronoi:
        current_image = apply_method3_voronoi(current_image, args.m3_num_points, args.m3_metric, args.m3_color_source)
        processed_suffix += "_m3"

    if args.method4_glyph_dither:
        current_image = apply_method4_glyph_dither(current_image, args.m4_num_colors, args.m4_glyph_size, 
                                                 args.m4_glyph_style, args.m4_base_glyph_color,
                                                 args.m4_use_quantized_color)
        processed_suffix += "_m4"

    if not processed_suffix:
        print("No processing methods selected. Nothing to do.")
        return

    output_filename = os.path.join(args.output_dir, f"{base_filename}{processed_suffix}.png")
    try:
        current_image.save(output_filename)
        print(f"Successfully saved processed image to: {output_filename}")
    except Exception as e:
        print(f"Error saving output image: {e}")

if __name__ == "__main__":
    main()
