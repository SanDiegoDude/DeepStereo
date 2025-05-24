import argparse
import os
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageChops, ImageColor # Added ImageChops
import numpy as np
import random
import math
from tqdm import tqdm
from datetime import datetime

# --- Helper Functions ---
# ... (parse_color, get_pixel_value_safe, resize_to_megapixels, rgb_to_hsv, hsv_to_rgb - unchanged)
def parse_color(color_str, default_color=(0, 0, 0)):
    try:
        if ',' in color_str: return tuple(map(int, color_str.split(',')))
    except ValueError: pass
    try: return ImageColor.getrgb(color_str)
    except ValueError: return default_color

def get_pixel_value_safe(pil_image, x, y):
    x = max(0, min(x, pil_image.width - 1)); y = max(0, min(y, pil_image.height - 1))
    return pil_image.getpixel((x, y))

def resize_to_megapixels(image_pil, target_mp):
    if target_mp <= 0: return image_pil
    w, h = image_pil.size; current_mp = (w * h) / 1_000_000.0
    if current_mp <= target_mp:
        print(f"Image already within target megapixels ({current_mp:.2f}MP <= {target_mp:.2f}MP). No resize.")
        return image_pil
    scale_factor = math.sqrt(target_mp / current_mp)
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    if new_w == 0 or new_h == 0:
        print(f"Warning: Calculated new dimensions are too small ({new_w}x{new_h}). Keeping original size."); return image_pil
    print(f"Resizing image from {w}x{h} ({current_mp:.2f}MP) to {new_w}x{new_h} (~{target_mp:.2f}MP)...")
    return image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

def rgb_to_hsv(r, g, b):
    r,g,b=r/255.0,g/255.0,b/255.0; mx=max(r,g,b); mn=min(r,g,b); df=mx-mn
    if mx==mn: h=0
    elif mx==r: h=(60*((g-b)/df)+360)%360
    elif mx==g: h=(60*((b-r)/df)+120)%360
    elif mx==b: h=(60*((r-g)/df)+240)%360
    if mx==0: s=0
    else: s=df/mx
    v=mx; return h,s,v

def hsv_to_rgb(h, s, v):
    i=math.floor(h/60)%6; f=(h/60)-math.floor(h/60); p=v*(1-s); q=v*(1-f*s); t=v*(1-(1-f)*s)
    if i==0: r,g,b=v,t,p
    elif i==1: r,g,b=q,v,p
    elif i==2: r,g,b=p,v,t
    elif i==3: r,g,b=p,q,v
    elif i==4: r,g,b=t,p,v
    elif i==5: r,g,b=v,p,q
    return int(r*255),int(g*255),int(b*255)

VIBRANT_PALETTE = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(255,128,0),(128,0,255)]

# --- Texture Generation Methods ---
# ... (apply_method1_color_dots, apply_method2_density_size, apply_method3_voronoi, apply_method4_glyph_dither - unchanged from previous version with tqdm)
def apply_method1_color_dots(content_image_pil, density=0.7, dot_size=1, bg_color=(0,0,0), color_mode="content_pixel", hue_shift_degrees=60):
    print(f"Applying Method 1: Color Dots (Density: {density}, Size: {dot_size}, Mode: {color_mode})")
    width,height=content_image_pil.size; output_image=Image.new("RGB",(width,height),bg_color); draw=ImageDraw.Draw(output_image)
    num_dots=int(width*height*density); content_rgb=content_image_pil.convert("RGB")
    for _ in tqdm(range(num_dots),desc="Method 1: Dots"):
        x,y=random.randint(0,width-1),random.randint(0,height-1); dot_color=(0,0,0)
        r_orig,g_orig,b_orig=get_pixel_value_safe(content_rgb,x,y)
        if color_mode=="content_pixel": dot_color=(r_orig,g_orig,b_orig)
        elif color_mode=="random_rgb": dot_color=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
        elif color_mode=="random_from_palette": dot_color=random.choice(VIBRANT_PALETTE)
        elif color_mode=="transformed_hue":
            h,s,v=rgb_to_hsv(r_orig,g_orig,b_orig); h=(h+random.uniform(-hue_shift_degrees,hue_shift_degrees))%360
            dot_color=hsv_to_rgb(h,s,v)
        elif color_mode=="transformed_invert": dot_color=(255-r_orig,255-g_orig,255-b_orig)
        if dot_size==1: draw.point((x,y),fill=dot_color)
        else: x0,y0=x-dot_size//2,y-dot_size//2; x1,y1=x0+dot_size-1,y0+dot_size-1; draw.rectangle([x0,y0,x1,y1],fill=dot_color)
    return output_image

def apply_method2_density_size(content_image_pil, mode="density", element_color=(0,0,0), bg_color=(255,255,255), base_size=2, max_size=10, invert_influence=False, density_factor=1.0):
    print(f"Applying Method 2: {mode.capitalize()} (Element: {element_color}, BG: {bg_color}, Invert: {invert_influence})")
    width,height=content_image_pil.size; output_image=Image.new("RGB",(width,height),bg_color); draw=ImageDraw.Draw(output_image)
    content_gray=content_image_pil.convert("L")
    if mode=="density":
        step=max(1,base_size)
        for r in tqdm(range(0,height,step),desc="Method 2: Density"):
            for c in range(0,width,step):
                brightness=get_pixel_value_safe(content_gray,c,r)/255.0; influence=brightness if invert_influence else 1.0-brightness
                if random.random()<(influence*density_factor): draw.rectangle([c,r,c+base_size-1,r+base_size-1],fill=element_color)
    elif mode=="size":
        num_elements=int((width*height*0.1)*density_factor)
        for _ in tqdm(range(num_elements),desc="Method 2: Size"):
            x,y=random.randint(0,width-1),random.randint(0,height-1); brightness=get_pixel_value_safe(content_gray,x,y)/255.0
            influence=brightness if invert_influence else 1.0-brightness
            current_size=max(1,int(base_size+(max_size-base_size)*influence))
            ex1,ey1=x-current_size//2,y-current_size//2; ex2,ey2=ex1+current_size-1,ey1+current_size-1
            draw.rectangle([ex1,ey1,ex2,ey2],fill=element_color)
    return output_image

def apply_method3_voronoi(content_image_pil, num_points=100, metric="F1", color_source="distance", point_color=(255,0,0)):
    print(f"Applying Method 3: Voronoi-like (Points: {num_points}, Metric: {metric}, Color: {color_source})")
    width,height=content_image_pil.size; output_image=Image.new("RGB",(width,height)); output_pixels=output_image.load()
    points=[]; content_rgb=content_image_pil.convert("RGB")
    for _ in range(num_points):
        px,py=random.randint(0,width-1),random.randint(0,height-1)
        p_content_color=get_pixel_value_safe(content_rgb,px,py); points.append((px,py,p_content_color))
    if not points: return content_image_pil.copy()
    max_dist_val=math.sqrt(width**2+height**2)/3
    for r in tqdm(range(height),desc="Method 3: Voronoi"):
        for c in range(width):
            distances_sq_data=[((c-px)**2+(r-py)**2,pcc) for px,py,pcc in points]; distances_sq_data.sort(key=lambda item:item[0])
            final_color=(0,0,0)
            if color_source=="distance":
                dist_val=0
                if metric=="F1" and len(distances_sq_data)>0: dist_val=math.sqrt(distances_sq_data[0][0])
                elif metric=="F2" and len(distances_sq_data)>1: dist_val=math.sqrt(distances_sq_data[1][0])
                elif metric=="F2-F1" and len(distances_sq_data)>1: dist_val=abs(math.sqrt(distances_sq_data[1][0])-math.sqrt(distances_sq_data[0][0]))
                norm_dist=min(dist_val/max_dist_val,1.0) if max_dist_val>0 else 0; gray_val=int(norm_dist*255)
                final_color=(gray_val,gray_val,gray_val)
            elif color_source=="content_point_color":
                idx=0;
                if metric=="F2" and len(distances_sq_data)>1: idx=1
                final_color=distances_sq_data[idx][1] if len(distances_sq_data)>idx else (0,0,0)
            elif color_source=="voronoi_cell_content_color":
                final_color=distances_sq_data[0][1] if distances_sq_data else (0,0,0)
            output_pixels[c,r]=final_color
    return output_image

def generate_glyph(glyph_style, size, glyph_element_color=(0,0,0), glyph_bg_color=(255,255,255)):
    glyph_img=Image.new("RGB",(size,size),glyph_bg_color); draw=ImageDraw.Draw(glyph_img)
    dot_density_factor=0.4
    if glyph_style=="random_dots":
        for _ in range(int(size*size*dot_density_factor)): draw.point((random.randint(0,size-1),random.randint(0,size-1)),fill=glyph_element_color)
    elif glyph_style=="lines":
        for i in range(0,size,max(1,size//3)): draw.line([(i,0),(i,size-1)],fill=glyph_element_color,width=max(1,size//8))
    elif glyph_style=="circles":
        padding=max(1,size//8); fill_color=glyph_bg_color if random.random()>0.6 else glyph_element_color
        draw.ellipse([(padding,padding,size-1-padding,size-1-padding)],outline=glyph_element_color,fill=fill_color,width=max(1,size//8))
    elif glyph_style=="solid": draw.rectangle([(0,0),(size-1,size-1)],fill=glyph_element_color)
    return glyph_img

def apply_method4_glyph_dither(content_image_pil, num_colors=8, glyph_size=8, glyph_style="random_dots", use_quantized_color_for_glyph_element=True):
    print(f"Applying Method 4: Glyph Dither (Colors: {num_colors}, Size: {glyph_size}, Style: {glyph_style})")
    width,height=content_image_pil.size; output_image=Image.new("RGB",(width,height),(128,128,128))
    try:
        quantized_content=content_image_pil.convert("RGB").quantize(colors=num_colors,method=Image.Quantize.MAXCOVERAGE)
        quantized_content_rgb=quantized_content.convert("RGB")
    except Exception as e:
        print(f"Warning: Quantization failed. Using original. Error: {e}"); quantized_content_rgb=content_image_pil.convert("RGB")
    for r_block in tqdm(range(0,height,glyph_size),desc="Method 4: Glyph Dither"):
        for c_block in range(0,width,glyph_size):
            block_center_x,block_center_y=min(c_block+glyph_size//2,width-1),min(r_block+glyph_size//2,height-1)
            glyph_main_color=get_pixel_value_safe(quantized_content_rgb,block_center_x,block_center_y)
            if not use_quantized_color_for_glyph_element: glyph_main_color=(0,0,0) if sum(glyph_main_color)>384 else (255,255,255)
            glyph_internal_bg_color=(0,0,0) if sum(glyph_main_color)>384 else (255,255,255)
            if glyph_style=="solid": glyph_internal_bg_color=glyph_main_color
            glyph=generate_glyph(glyph_style,glyph_size,glyph_element_color=glyph_main_color,glyph_bg_color=glyph_internal_bg_color)
            output_image.paste(glyph,(c_block,r_block))
    return output_image


# --- Image Blending Functions ---
def blend_images(base_image, blend_image, mode="average", opacity=1.0):
    """ Blends two images using a specified mode. Base image is the first, blend_image is applied on top."""
    if base_image.size != blend_image.size or base_image.mode != blend_image.mode:
        # Attempt to convert if modes differ but ensure they are RGB for chops
        blend_image = blend_image.convert(base_image.mode)
        if base_image.size != blend_image.size:
             raise ValueError("Images must be the same size to blend.")
    
    # Ensure images are RGB for ImageChops compatibility if needed
    base_rgb = base_image.convert("RGB")
    blend_rgb = blend_image.convert("RGB")

    result_rgb = None

    if mode == "average":
        # Image.blend is alpha blending. For true average:
        # (base * (1-opacity) + blend * opacity) - this is more like photoshop opacity slider.
        # For a 50/50 average of the two, opacity would effectively be 0.5 for the 'blend' image.
        # Image.blend(im1, im2, alpha) -> out = im1 * (1.0 - alpha) + im2 * alpha
        # So if we want (base + blend)/2, it's blend(base, blend, 0.5)
        result_rgb = Image.blend(base_rgb, blend_rgb, 0.5) # True average
    elif mode == "lighten":
        result_rgb = ImageChops.lighter(base_rgb, blend_rgb)
    elif mode == "darken":
        result_rgb = ImageChops.darker(base_rgb, blend_rgb)
    elif mode == "multiply":
        result_rgb = ImageChops.multiply(base_rgb, blend_rgb)
    elif mode == "screen":
        result_rgb = ImageChops.screen(base_rgb, blend_rgb)
    elif mode == "add":
        result_rgb = ImageChops.add(base_rgb, blend_rgb)
    elif mode == "difference":
        result_rgb = ImageChops.difference(base_rgb, blend_rgb)
    elif mode == "overlay":
        # Pillow's ImageChops doesn't have overlay directly.
        # We need to implement it or use a more complex approach (e.g. numpy)
        # For simplicity, we can approximate or use a similar mode.
        # True overlay: if base_pixel < 0.5: 2 * base * blend else: 1 - 2 * (1-base) * (1-blend)
        # Using numpy for this:
        base_arr = np.array(base_rgb, dtype=float) / 255.0
        blend_arr = np.array(blend_rgb, dtype=float) / 255.0
        
        # Create an empty array for the result
        overlay_arr = np.zeros_like(base_arr)
        
        # Condition for overlay
        # Where base_arr <= 0.5, result is 2 * base_arr * blend_arr
        # Where base_arr > 0.5, result is 1 - 2 * (1 - base_arr) * (1 - blend_arr)
        low_mask = base_arr <= 0.5
        high_mask = ~low_mask # Invert the mask for values > 0.5
        
        overlay_arr[low_mask] = 2 * base_arr[low_mask] * blend_arr[low_mask]
        overlay_arr[high_mask] = 1 - 2 * (1 - base_arr[high_mask]) * (1 - blend_arr[high_mask])
        
        result_rgb = Image.fromarray((np.clip(overlay_arr, 0, 1) * 255).astype(np.uint8), "RGB")

    else: # Fallback to base if mode is unknown
        result_rgb = base_rgb
    
    # Handle opacity if an actual blend occurred and opacity is not 1.0
    if opacity < 1.0 and result_rgb != base_rgb :
        # Blend the result with the original base_image based on opacity
        # result_rgb = Image.blend(base_image.convert("RGB"), result_rgb, opacity) # This blends result ONTO base.
                                                                                 # We want the computed result_rgb to be faded TOWARDS base.
        # If opacity is how much of the BLEND_IMAGE's effect is shown.
        # Result_from_blend_mode is effectively the "blend_image" at 100% opacity for that mode.
        # Now, we want to reduce its effect.
        # Final = base * (1-opacity) + Result_from_blend_mode * opacity
        final_blend = Image.blend(base_image.convert("RGB"), result_rgb, opacity) # opacity applies to result_rgb
        return final_blend

    return result_rgb


# --- Main Execution ---
def main():
    # ... (previous main function setup - argparse)
    print("--- Procedural Texture Generator ---")
    print("NOTE: All processing is done on the CPU and can be slow for large images or complex methods.")
    parser = argparse.ArgumentParser(description="Procedurally generate textures influenced by a content image.")
    parser.add_argument("--input", required=True, help="Path to the input content image.")
    parser.add_argument("--output_dir", default="output_textures", help="Directory to save the processed image.")
    parser.add_argument("--max_megapixels", type=float, default=1.0, help="Resize input image to approx this many megapixels before processing (e.g., 1.0 for 1MP). 0 for no resize. (default: 1.0)")
    
    parser.add_argument("--combination_mode", choices=["sequential", "blend"], default="sequential",
                        help="How to combine outputs if multiple methods are selected. (default: sequential)")
    parser.add_argument("--blend_type", choices=["average", "lighten", "darken", "multiply", "screen", "add", "difference", "overlay"],
                        default="overlay", help="Blend mode to use if combination_mode is 'blend'. (default: overlay)")
    parser.add_argument("--blend_opacity", type=float, default=1.0, help="Opacity for each blend step when combination_mode is 'blend' (0.0-1.0). Default: 1.0")


    # Method Args (no changes to these arg definitions)
    # Method 1 Args
    parser.add_argument("--method1_color_dots", action="store_true", help="Apply Method 1: Content-driven color dots.")
    parser.add_argument("--m1_density", type=float, default=0.7, help="Dot density for Method 1 (0.0 to 1.0).")
    parser.add_argument("--m1_dot_size", type=int, default=2, help="Dot size for Method 1 (pixels).")
    parser.add_argument("--m1_bg_color", type=str, default="black", help="Background color for Method 1 (R,G,B or name).")
    parser.add_argument("--m1_color_mode", choices=["content_pixel", "random_rgb", "random_from_palette", "transformed_hue", "transformed_invert"], default="content_pixel", help="Color mode for dots in Method 1.")
    parser.add_argument("--m1_hue_shift_degrees", type=float, default=90, help="Max +/- hue shift for 'transformed_hue' mode (degrees).")
    # Method 2 Args
    parser.add_argument("--method2_density_size", action="store_true", help="Apply Method 2: Content-driven density or size of elements.")
    parser.add_argument("--m2_mode", choices=["density", "size"], default="density", help="Mode for Method 2.")
    parser.add_argument("--m2_element_color", type=str, default="white", help="Element color for Method 2 (R,G,B or name).")
    parser.add_argument("--m2_bg_color", type=str, default="black", help="Background color for Method 2 (R,G,B or name).")
    parser.add_argument("--m2_base_size", type=int, default=3, help="Base size for elements in Method 2.")
    parser.add_argument("--m2_max_size", type=int, default=12, help="Max size for elements in Method 2 (size mode).")
    parser.add_argument("--m2_invert_influence", action="store_true", help="Invert content brightness influence for Method 2.")
    parser.add_argument("--m2_density_factor", type=float, default=1.0, help="Overall density factor for Method 2 (higher means more elements).")
    # Method 3 Args
    parser.add_argument("--method3_voronoi", action="store_true", help="Apply Method 3: Simplified Voronoi/Worley-like noise.")
    parser.add_argument("--m3_num_points", type=int, default=200, help="Number of seed points for Method 3.")
    parser.add_argument("--m3_metric", choices=["F1", "F2", "F2-F1"], default="F1", help="Distance metric for Method 3.")
    parser.add_argument("--m3_color_source", choices=["distance", "content_point_color", "voronoi_cell_content_color"], default="distance", help="Color source for Method 3 cells. 'voronoi_cell_content_color' uses the color of the closest point's original content.")
    # Method 4 Args
    parser.add_argument("--method4_glyph_dither", action="store_true", help="Apply Method 4: Stylized dithering with glyphs.")
    parser.add_argument("--m4_num_colors", type=int, default=8, help="Number of colors for quantization in Method 4.")
    parser.add_argument("--m4_glyph_size", type=int, default=10, help="Size of glyphs for Method 4 (pixels).")
    parser.add_argument("--m4_glyph_style", choices=["random_dots", "lines", "circles", "solid"], default="random_dots", help="Style of glyphs for Method 4.")
    parser.add_argument("--m4_use_quantized_color_for_glyph_element", action="store_true", help="Use quantized color from content image for glyph's main elements. If false, glyph elements use a fixed contrasting color.")

    args = parser.parse_args()

    # ... (initial setup: input check, output dir, filename, timestamp, image loading - unchanged)
    if not os.path.exists(args.input): print(f"Error: Input image '{args.input}' not found."); return
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir, exist_ok=True); print(f"Creating output directory: {args.output_dir}")
    base_filename = os.path.splitext(os.path.basename(args.input))[0]; timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        initial_image_loaded = Image.open(args.input)
        print(f"Loaded input image: {args.input} (Original Size: {initial_image_loaded.size})")
    except Exception as e: print(f"Error loading input image: {e}"); return

    # --- Apply Resize ---
    processed_suffix = ""; initial_processed_image = initial_image_loaded.copy() # Start with a copy
    if args.max_megapixels > 0:
        resized_image = resize_to_megapixels(initial_processed_image, args.max_megapixels)
        if resized_image.size != initial_processed_image.size:
            processed_suffix += f"_resized{args.max_megapixels:.1f}MP"
        initial_processed_image = resized_image
        print(f"Image for processing: {initial_processed_image.size}.")
    else:
        print("No resizing requested based on megapixels.")

    method_outputs = []
    methods_to_run = []

    if args.method1_color_dots: methods_to_run.append(lambda img: apply_method1_color_dots(img, args.m1_density, args.m1_dot_size, parse_color(args.m1_bg_color), args.m1_color_mode, args.m1_hue_shift_degrees))
    if args.method2_density_size: methods_to_run.append(lambda img: apply_method2_density_size(img, args.m2_mode, parse_color(args.m2_element_color), parse_color(args.m2_bg_color), args.m2_base_size, args.m2_max_size, args.m2_invert_influence, args.m2_density_factor))
    if args.method3_voronoi: methods_to_run.append(lambda img: apply_method3_voronoi(img, args.m3_num_points, args.m3_metric, args.m3_color_source))
    if args.method4_glyph_dither: methods_to_run.append(lambda img: apply_method4_glyph_dither(img, args.m4_num_colors, args.m4_glyph_size, args.m4_glyph_style, args.m4_use_quantized_color_for_glyph_element))

    if not methods_to_run:
        if processed_suffix: # Only resize was done
            print(f"Only resizing was applied. Saving resized image.")
            output_filename = os.path.join(args.output_dir, f"{base_filename}{processed_suffix}_{timestamp}.png")
            try: initial_processed_image.save(output_filename); print(f"Successfully saved to: {output_filename}")
            except Exception as e: print(f"Error saving: {e}")
        else: print("No processing methods selected, and no resize. Nothing to do.")
        return

    final_image = None

    if args.combination_mode == "sequential":
        print("Applying methods sequentially...")
        current_image_for_seq = initial_processed_image.copy()
        method_name_map = {0:"_m1", 1:"_m2", 2:"_m3", 3:"_m4"} # Assuming order of adding to methods_to_run
        actual_methods_run_indices = [] # Track which methods actually get selected by args
        if args.method1_color_dots: actual_methods_run_indices.append(0)
        if args.method2_density_size: actual_methods_run_indices.append(1)
        if args.method3_voronoi: actual_methods_run_indices.append(2)
        if args.method4_glyph_dither: actual_methods_run_indices.append(3)

        for i, method_func_idx in enumerate(actual_methods_run_indices):
            # Find the actual method function from the original methods_to_run list
            # This is a bit convoluted because methods_to_run is populated by lambdas
            # A better way would be to associate names with functions earlier
            # For now, let's assume methods_to_run directly corresponds to the args.methodX flags
            # This needs to be more robust if method order can change.
            # Let's rebuild methods_to_run based on actual args for sequential
            
            current_image_for_seq = methods_to_run[i](current_image_for_seq) #This assumes methods_to_run only contains selected methods
            processed_suffix += method_name_map.get(method_func_idx, f"_m{method_func_idx+1}")


        final_image = current_image_for_seq

    elif args.combination_mode == "blend":
        print(f"Applying methods individually for blending (Mode: {args.blend_type}, Opacity: {args.blend_opacity})...")
        if not methods_to_run:
             print("No methods selected for blending. Using initial image.")
             final_image = initial_processed_image
        else:
            for i, method_func in enumerate(methods_to_run):
                print(f"Running individual method {i+1}/{len(methods_to_run)}...")
                # Pass a copy of the *initial_processed_image* (original, resized) to each method
                method_output = method_func(initial_processed_image.copy())
                method_outputs.append(method_output)
                # Add to suffix based on which arg was true
                if method_func.__closure__[0].cell_contents.__name__.startswith("apply_method1"): processed_suffix += "_m1"
                elif method_func.__closure__[0].cell_contents.__name__.startswith("apply_method2"): processed_suffix += "_m2"
                elif method_func.__closure__[0].cell_contents.__name__.startswith("apply_method3"): processed_suffix += "_m3"
                elif method_func.__closure__[0].cell_contents.__name__.startswith("apply_method4"): processed_suffix += "_m4"


            if not method_outputs:
                final_image = initial_processed_image # Should not happen if methods_to_run was populated
            else:
                final_image = method_outputs[0] # Start with the first method's output
                if len(method_outputs) > 1:
                    for i in range(1, len(method_outputs)):
                        print(f"Blending result with output of method {i+1} using '{args.blend_type}'...")
                        final_image = blend_images(final_image, method_outputs[i], mode=args.blend_type, opacity=args.blend_opacity)
        processed_suffix += f"_blend_{args.blend_type}"


    if final_image is None: # Should only happen if only resize was done and no methods
        if processed_suffix: # only resize
             final_image = initial_processed_image
        else:
            print("No methods run and no effective resize. Nothing to save."); return


    output_filename = os.path.join(args.output_dir, f"{base_filename}{processed_suffix}_{timestamp}.png")
    try:
        final_image.save(output_filename)
        print(f"Successfully saved processed image to: {output_filename}")
    except Exception as e:
        print(f"Error saving output image: {e}")

if __name__ == "__main__":
    main()
