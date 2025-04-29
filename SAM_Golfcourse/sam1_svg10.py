#!/home/chuck/venv/bin/python3

import numpy as np
import cv2
from svgwrite import Drawing
import base64
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
import argparse
import matplotlib.pyplot as plt

# --- Keep other imports ---
import os
import sys
sys.path.append("..")
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F # Often import functional as F
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    print("Error: segment_anything library not found.")

# import the models.py for ClassificaonHead
from models import ClassificationHead

# --- Global Configuration Constants ---
# Define the size to which the input image is initially resized before SAM processing.
# This is done to manage GPU memory based on the specific SAM model and GPU VRAM.
# Format is (width, height) as expected by cv2.resize. 1892 worked for RTX 4060 8GB.
# If your VRAM is larger try 2048x2048 or larger.  Your may need to trial-n-error this.
IMAGE_INPUT_SIZE = (1792, 1792) # 1024 + 512 + 256.

# --- This reduces the number of control points from the curve fitting below.
# --- based on the Ramer-Douglas-Peucker (RDP) algorithm. Epsilon is the control value. 5.0 is agressive, 0.5 is minimal. 
def rdp_vec(points, epsilon):
    """
    Simplifies a list of points using the Ramer-Douglas-Peucker algorithm (vectorized).

    Args:
        points: A numpy array of shape (N, 2) representing (x, y) coordinates.
        epsilon: The maximum distance threshold.

    Returns:
        A numpy array of shape (M, 2) representing the simplified polyline, where M <= N.
    """
    if points.shape[0] < 3:
        return points

    # Find the point with the maximum distance using vectorized operations
    start_point = points[0]
    end_point = points[-1]

    # Vector from start to end
    line_vec = end_point - start_point
    line_len_sq = np.sum(line_vec**2)

    # Vector from start to each intermediate point
    point_vec = points[1:-1] - start_point

    # Project intermediate points onto the line segment
    # Handle zero-length line segment
    if line_len_sq == 0:
        # All points are coincident with start/end, distance is to start_point
        dists = np.sqrt(np.sum((points[1:-1] - start_point)**2, axis=1))
        # Set projection parameter `t` to 0 to indicate closest point is start_point
        t = np.zeros(points.shape[0] - 2)
    else:
        t = np.dot(point_vec, line_vec) / line_len_sq
        t = np.clip(t, 0, 1) # Clamp projection to be within the segment

        # Calculate projected points on the line segment
        projected_points = start_point + t[:, np.newaxis] * line_vec

        # Calculate perpendicular distances from intermediate points to the segment
        dists = np.sqrt(np.sum((points[1:-1] - projected_points)**2, axis=1))

    # Find the index of the point with maximum distance
    max_dist_index = np.argmax(dists)
    max_dist = dists[max_dist_index]

    # If max distance is greater than epsilon, recursively simplify
    if max_dist > epsilon:
        # Recursive calls (indices need to be adjusted back to original array)
        rec_results1 = rdp_vec(points[:max_dist_index + 2], epsilon) # Include the splitting point
        rec_results2 = rdp_vec(points[max_dist_index + 1:], epsilon) # Start from splitting point

        # Combine results (avoiding duplicate splitting point)
        simplified_points = np.vstack((rec_results1[:-1], rec_results2))
    else:
        # If distance threshold not met, the segment is the start and end points
        simplified_points = np.vstack((points[0], points[-1]))

    return simplified_points
# --- MODIFIED Function with RDP Simplification and Class Coloring ---
def show_anns_with_contours(anns, image, output_path=None,
                            outline_color='lime',
                            smoothing_method='gaussian',
                            smoothing_s_factor=500.0,
                            smoothing_sigma=2.0,
                            num_spline_points=300,
                            mask_alpha=0.5,
                            rdp_epsilon=1.5,
                            class_colors_bgr=None): # <<< ADDED PARAMETER for class colors
    """
    Generates an SVG file with simplified, smoothed, filled/outlined paths,
    colored based on predicted class ID.
    Applies RDP simplification after Gaussian smoothing if rdp_epsilon > 0.

    Args:
        # ... (existing args) ...
        rdp_epsilon: RDP simplification tolerance (pixels). 0 disables.
        class_colors_bgr: Dictionary mapping predicted class ID (int) to BGR tuple (int, int, int). <<< ADDED
    """
    if not anns: print("No annotations found."); return

    # --- Define default colors if none provided ---
    if class_colors_bgr is None:
         print("Warning: No class_colors_bgr provided. Using random colors or default fallback.")
         # Use a default mapping or random color if needed
         # For now, we'll rely on the classification logic providing 'predicted_class_id'
         # and the loop below handling missing colors gracefully.
         # The random color logic was removed for clarity.

    # ... (Annotation processing - modified to use predicted_class_id and specific color) ...
    # The input 'anns' list is now expected to contain dicts with 'predicted_class_id'

    try:
        # Still sort by area if 'area' is present in the mask dicts
        # SamAutomaticMaskGenerator output has 'area', so this should work.
        sorted_anns = sorted(anns, key=(lambda x: x.get('area', 0)), reverse=True) # Use .get for safety
    except Exception as e:
        print(f"Warning: Could not sort annotations by area ({e}). Using unsorted annotations.")
        sorted_anns = anns


    height, width, channels = image.shape
    # ... (image_bgr handling - same as before) ...
    if channels == 4:
        print("Converting input BGRA image to BGR.")
        image_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif channels == 1:
        print("Converting input Grayscale image to BGR.")
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif channels == 3:
        print("Input image is BGR. Using a copy.")
        image_bgr = image.copy()
    else:
        print(f"Error: Unexpected number of image channels: {channels}")
        return

    # --- (blended_preview calculation - optional, mostly for debugging) ---
    # Keep this part as it helps visualize the masks on the image
    blended_preview = image_bgr.copy()
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    alpha_byte = int(np.clip(mask_alpha * 255, 0, 255))
    print(f"Generating blended preview for {len(sorted_anns)} masks with alpha={mask_alpha:.2f}...")

    for ann_data in sorted_anns:
        m = ann_data['segmentation']
        predicted_class_id = ann_data.get('predicted_class_id', 0) # Get predicted class, default to 0 (background)

        # Select color for preview (using the same color map as SVG)
        preview_color_bgr = class_colors_bgr.get(predicted_class_id, (0, 0, 0)) # Default black if class ID not in map

        if m.shape != (height, width):
             # This resize warning is redundant if masks are already resized after generation, but keep as safeguard
             print(f"Warning: Resizing mask inside show_anns from {m.shape} to {(height, width)} for preview")
             m = cv2.resize(m.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)

        overlay[m, 0] = int(preview_color_bgr[0])
        overlay[m, 1] = int(preview_color_bgr[1])
        overlay[m, 2] = int(preview_color_bgr[2])
        overlay[m, 3] = alpha_byte # Apply alpha

    alpha_normalized = overlay[:, :, 3] / 255.0
    alpha_3c = np.stack([alpha_normalized]*3, axis=-1)
    overlay_bgr = overlay[:, :, :3]
    blended_float = overlay_bgr.astype(float) * alpha_3c + \
                    blended_preview.astype(float) * (1.0 - alpha_3c)
    blended_preview = blended_float.astype(np.uint8)


    if output_path:
        print(f"Creating SVG image size Width: {width} Height: {height}")
        dwg = Drawing(filename=output_path, size=(width, height))
        # Determine the path for the background image file
        # We'll save it next to the SVG with a '_background.png' suffix
        background_filename = os.path.splitext(os.path.basename(output_path))[0] + "_background.png"
        background_filepath = os.path.join(os.path.dirname(output_path), background_filename)

        print(f"Saving background image to: {background_filepath}")
        # Save the BGR image (used for blending preview and desired background) as a PNG
        success = cv2.imwrite(background_filepath, image_bgr)

        if success:
            print("Background image saved successfully.")
            # Add the image to the SVG, linking to the external file
            # Use the relative path from the SVG file to the background file
            dwg.add(dwg.image(background_filename, # Use just the filename for the link
                              insert=(0, 0),
                              size=(width, height)))
            print(f"Linked external background image '{background_filename}' in SVG.")
        else:
            print(f"Error: Failed to save background image to {background_filepath}. SVG will have no background.")
            # Optionally, you could fall back to embedding the Base64 here,
            # but let's keep it simple and rely on external linking for now.

        print(f"Processing {len(sorted_anns)} annotations for SVG (Smoothing: {smoothing_method}, RDP Eps: {rdp_epsilon})...")
        contour_count = 0
        total_original_points = 0
        total_simplified_points = 0

        for i, ann_data in enumerate(sorted_anns):
            mask = ann_data['segmentation'].astype(np.uint8) # Use the (potentially resized) mask from ann_data
            predicted_class_id = ann_data.get('predicted_class_id', 0) # Get predicted class

            # --- Select FILL color based on predicted class ID ---
            if class_colors_bgr and predicted_class_id in class_colors_bgr:
                 fill_color_bgr = class_colors_bgr[predicted_class_id]
                 # Convert BGR tuple to Hex string
                 fill_color_hex = "#{:02x}{:02x}{:02x}".format(int(fill_color_bgr[2]), int(fill_color_bgr[1]), int(fill_color_bgr[0])) # BGR to RGB hex
            else:
                 # Fallback if class_colors_bgr is None or class ID not found
                 fill_color_hex = "#000000" # Default to black
                 if class_colors_bgr is not None:
                      print(f"Warning: No color defined for class ID {predicted_class_id}. Using default black for contour {contour_count}.")

            fill_alpha = mask_alpha
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for contour in contours:
                if len(contour) < 4: continue # Skip very small contours

                original_contour_points = contour.reshape(-1, 2).astype(float)
                x = original_contour_points[:, 0]
                y = original_contour_points[:, 1]
                num_orig = len(x)
                total_original_points += num_orig

                try:
                    final_points_for_svg = None # Store the points to draw

                    if smoothing_method == 'spline':
                        # ... (Spline smoothing code - same as before) ...
                        x_spline = np.r_[x, x[0]]; y_spline = np.r_[y, y[0]]
                        if len(x_spline) < 4: continue # Need at least 4 points for cubic spline
                        # Ensure s is not too large for short contours
                        max_s = len(x_spline)**2
                        s_factor = min(smoothing_s_factor, max_s * 0.5) # Cap s_factor to avoid errors on small contours

                        tck, u = splprep([x_spline, y_spline], s=s_factor, k=min(3, len(x_spline)-1), per=True) # k=3, but reduce k if contour too short
                        u_new = np.linspace(u.min(), u.max(), num_spline_points)
                        x_smooth, y_smooth = splev(u_new, tck, der=0)

                        final_points_for_svg = np.vstack((x_smooth, y_smooth)).T
                        print(f"  Ann {i+1}, Contour: Spline smoothed to {len(final_points_for_svg)} points.")

                    elif smoothing_method == 'gaussian':
                        if num_orig < 1: continue
                        x_smooth = gaussian_filter1d(x, sigma=smoothing_sigma, mode='wrap')
                        y_smooth = gaussian_filter1d(y, sigma=smoothing_sigma, mode='wrap')
                        smoothed_points = np.vstack((x_smooth, y_smooth)).T

                        # <<< APPLY RDP SIMPLIFICATION >>>
                        if rdp_epsilon > 0 and len(smoothed_points) >= 3: # RDP needs at least 3 points
                            simplified_points = rdp_vec(smoothed_points, rdp_epsilon)
                            num_simp = len(simplified_points)

                            # Ensure RDP didn't collapse it too much (< 3 points for a polygon)
                            if num_simp < 3:
                                print(f"  Warning: RDP resulted in < 3 points ({num_simp}). Using original smoothed points for this contour.")
                                final_points_for_svg = smoothed_points
                                total_simplified_points += len(smoothed_points) # Count original smoothed points
                            else:
                                final_points_for_svg = simplified_points
                                total_simplified_points += num_simp # Count simplified points
                            print(f"  Ann {i+1}, Contour: Gaussian smoothed, RDP {num_orig} -> {num_simp} points (eps={rdp_epsilon})")

                        else:
                            # No RDP simplification applied
                            final_points_for_svg = smoothed_points
                            total_simplified_points += len(smoothed_points) # Count original smoothed points
                            # Print message based on why RDP wasn't applied
                            if rdp_epsilon <= 0:
                                print(f"  Ann {i+1}, Contour: Gaussian smoothed ({num_orig} points), RDP disabled (epsilon <= 0).")
                            else: # rdp_epsilon > 0 but num_orig < 3
                                print(f"  Ann {i+1}, Contour: Gaussian smoothed ({num_orig} points), too few points for RDP.")


                    else: # Fallback: No smoothing
                        final_points_for_svg = original_contour_points
                        total_simplified_points += len(final_points_for_svg) # Count original points
                        print(f"  Ann {i+1}, Contour: No smoothing applied ({num_orig} points).")


                    # *** Generate SVG path from final_points_for_svg ***
                    if final_points_for_svg is not None and len(final_points_for_svg) >= 3: # Need at least 3 points for a polygon path
                        # Format points for SVG path data (M = moveto, L = lineto, Z = closepath)
                        # M X0,Y0 L X1,Y1 L X2,Y2 ... L Xn,Yn Z
                        path_data = "M " + " ".join([f"{p[0]:.2f},{p[1]:.2f}" for p in final_points_for_svg]) + " Z"
                        path = dwg.path(d=path_data,
                                         fill=fill_color_hex, # Use the class color
                                         fill_opacity=mask_alpha,
                                         stroke=outline_color,
                                         stroke_width=1)
                        dwg.add(path)
                        contour_count += 1
                    elif final_points_for_svg is not None and len(final_points_for_svg) > 0:
                         print(f"  Warning: Contour {contour_count+1} resulted in < 3 points after simplification. Skipping SVG path creation.")
                         # You could draw circles for single points if needed, but usually skip.

                except Exception as e:
                    print(f"  Error processing contour in annotation {i+1} (Orig Pts: {num_orig}): {e}")


        print(f"Added {contour_count} filled/stroked path elements to SVG.")
        print(f"Total points before simplification (approx): {total_original_points}")
        print(f"Total points after simplification (approx): {total_simplified_points}")
        dwg.save()
        print(f"SVG file saved to: {output_path}")
        print(">>> RDP Epsilon controls simplification level for Gaussian smoothing. <<<")
        print(">>> Tune --rdp_epsilon (try values like 0.5, 1.0, 2.0, 5.0) to adjust point reduction. <<<")

    else:
        print("Displaying blended preview image (no SVG saved).")
        plt.figure(figsize=(15, 15))
        # Matplotlib expects RGB, but OpenCV reads BGR. Convert for plotting.
        plt.imshow(cv2.cvtColor(blended_preview, cv2.COLOR_BGR2RGB))
        plt.title("Blended Mask Preview (SVG not saved)")
        plt.axis('off')
        plt.show()



def main():

    # --- Define a color mapping for your classes (BGR format for OpenCV/svwrite) ---
    # Match the order and IDs from your dataset/training (0=background, 1=Fairway, 2=Green, etc.)
    # You might want to make this configurable via args or a config file later
    class_colors_bgr = {
        0: (0, 0, 0),       # Class 0: Background (or not drawn) - Black BGR
        1: (144, 238, 144), # Class 1: Fairway - Light Green BGR
        2: (0, 100, 0),     # Class 2: Green - Dark Green BGR
        3: (210, 180, 140), # Class 3: Bunker - Tan BGR
        4: (173, 216, 230), # Class 4: Water - Light Blue BGR
        5: (128, 128, 128), # Class 5: Rough - Grey BGR
        # Add more classes if you have them
    }
    num_classes = len(class_colors_bgr) # Determine num_classes from your color map

    cwd = os.getcwd()
    home = os.path.expanduser("~") # Use os.path.expanduser("~") or pathlib.Path.home()

    parser = argparse.ArgumentParser(description="Generate SVG with filled/outlined mask paths from SAM segmentations, with optional simplification and classification coloring.")
    # ... (Input/Output Arguments - same as before) ...
    parser.add_argument("input_image", help="Path to the input image.")
    parser.add_argument("-o", "--output", help="Path to the output SVG file (optional). If not provided, plot is shown.")
    parser.add_argument("--outline", default="lime", help="Color of the contour outlines. Default: lime")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha transparency for the filled SVG mask paths. Default: 0.5")

    # Update default checkpoint paths to use the user's home directory or relative paths consistently
    # Using relative paths from the script location might be better if you version control 'training' folder
    parser.add_argument("--sam_checkpoint", default=os.path.join(cwd, "training", "sam_vit_b_01ec64.pth"), help="Path to the original SAM checkpoint (e.g., vit_b).")
    parser.add_argument("--sam_trained_checkpoint", default=os.path.join(cwd, "training", "sam_finetuned_golf_epoch_10.pth"), help="Path to your fine-tuned SAM checkpoint.") # Corrected default filename
    parser.add_argument("--model_type", default="vit_b", choices=["vit_h", "vit_l", "vit_b"], help="SAM model type used for training.")
    parser.add_argument("--device", default="cuda", help="Device for SAM model (e.g., 'cuda', 'cuda:0', 'cpu').") # Default to 'cuda'

    # --- SAM Automatic Mask Generator Parameters ---
    # Note: Automatic Mask Generator uses its own parameters for proposing masks,
    # separate from the point/box prompts used during training/prediction.
    # These parameters control the density and quality of the generated initial masks.
    # You might need to tune these for best results on your images.
    parser.add_argument("--points_per_side", type=int, default=32, help="SAM points per side for grid sampling in Automatic Mask Generator.")
    parser.add_argument("--pred_iou_thresh", type=float, default=0.88, help="[AMG] Predicted IoU threshold for filtering masks. Default: 0.88 (often higher than 0.86)")
    parser.add_argument("--stability_score_thresh", type=float, default=0.92, help="[AMG] Mask stability score threshold for filtering masks. Default: 0.92")
    parser.add_argument("--min_mask_region_area", type=int, default=100, help="[AMG] Minimum mask region area in pixels (at input image scale) for filtering masks.")
    # --- End AMG Parameters ---


    # ... (Smoothing/Simplification Arguments - same as before) ...
    parser.add_argument("--smoothing_method", default="gaussian", choices=["spline", "gaussian"], help="Smoothing method ('spline' or 'gaussian'). Default: gaussian")
    parser.add_argument("--smoothing_s", type=float, default=500.0, help="[spline method] Smoothing factor 's'. Default: 500.0")
    parser.add_argument("--smoothing_sigma", type=float, default=2.0, help="[gaussian method] Sigma for gaussian_filter1d. Default: 2.0")
    parser.add_argument("--spline_points", type=int, default=300, help="Number of points for final SVG path (spline resampling). Default: 300")
    # --- Control point reduction
    parser.add_argument("--rdp_epsilon", type=float, default=1.5, help="Epsilon for Ramer-Douglas-Peucker simplification after smoothing. Higher values mean more simplification (fewer points). Set to 0 to disable. Default: 1.5.")


    args = parser.parse_args()


    # --- (Image Loading and Initial Resize) ---
    input_image_path = args.input_image
    if not os.path.exists(input_image_path):
        print(f"Error: Input image '{input_image_path}' not found.")
        sys.exit(1)
    print(f"Loading image: {input_image_path}")
    original_image = cv2.imread(input_image_path) # Load as BGR
    if original_image is None:
        print(f"Error: Failed to load image '{input_image_path}'.")
        sys.exit(1)

    original_height, original_width = original_image.shape[:2]
    print(f"Original image dimensions: {original_width}x{original_height}")

    # --- Image Preprocessing for SAM (Resize Longest Side + Pad) ---
    # SAM's internal preprocessing handles resizing to 1024 longest side and padding.
    # We should feed it the image *after* any initial downscaling if needed for memory,
    # but let SAM's internal transform handle the final 1024 sizing.
    # The original image size (original_width, original_height) is needed for postprocessing/SVG.

    # Let's keep your 1892 resize for now, but note the comment about it being an extra step.
    print(f"Initially resizing image to {IMAGE_INPUT_SIZE[0]}x{IMAGE_INPUT_SIZE[1]} #for memory considerations.")
    # Use INTER_AREA for shrinking, it's generally preferred
    initial_resized_image = cv2.resize(original_image, IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA)

    # SAM expects RGB input (HWC)
    channels = initial_resized_image.shape[2]
    if channels == 4:
        image_rgb = cv2.cvtColor(initial_resized_image, cv2.COLOR_BGRA2RGB)
    elif channels == 1:
        image_rgb = cv2.cvtColor(initial_resized_image, cv2.COLOR_GRAY2RGB)
    elif channels == 3:
        image_rgb = cv2.cvtColor(initial_resized_image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    else:
        print(f"Error: Unexpected number of image channels after initial resize: {channels}")
        sys.exit(1)


    print("Loading SAM model...")
    try:
        # --- Load the base SAM model with original checkpoint ---
        # This loads the image encoder, prompt encoder, mask decoder weights from the official release
        if not os.path.exists(args.sam_checkpoint):
            print(f"Error: Original SAM checkpoint not found at '{args.sam_checkpoint}'")
            sys.exit(1) # Exit if base checkpoint is missing

        sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
        sam.eval() # Set base SAM to evaluation mode

        # --- Set Device ---
        device_str = args.device
        if "cuda" in device_str:
            if not torch.cuda.is_available():
                device_str = "cpu"
                print("Warning: CUDA specified but not available via torch. Using CPU.")
            else:
                 # Check if specific CUDA device is available
                 try:
                     device = torch.device(device_str)
                     torch.cuda.get_device_name(device) # This will raise error if device index is invalid
                 except Exception:
                     print(f"Warning: CUDA device '{device_str}' not available. Using 'cuda' default.")
                     device_str = "cuda" # Fallback to default CUDA device


        device = torch.device(device_str)
        sam.to(device=device) # Move SAM model to device

        if "cpu" in device_str:
            # Set CPU threads if running on CPU
            # You have 16 cores, 30 might be aggressive if other things run, try number of physical cores
            # Or just let torch manage defaults
            # torch.set_num_threads(16) # Or your preference

            print("Warning: Running on CPU. This will be significantly slower than GPU.")


        # --- Load and Apply the Trained Checkpoint (fine-tuned weights) ---
        # This replaces the prompt encoder and mask decoder weights, and includes classification head weights
        if os.path.exists(args.sam_trained_checkpoint):
            print(f"Loading fine-tuned checkpoint from '{args.sam_trained_checkpoint}'")
            trained_state_dict = torch.load(args.sam_trained_checkpoint, map_location=device) # Load checkpoint to the correct device

            # --- Load the fine-tuned state_dict into the sam model (strict=False) ---
            # This updates the prompt encoder and mask decoder weights in the 'sam' instance.
            # Use strict=False because the trained_state_dict does NOT have image encoder keys
            # and DOES have classification head keys that don't belong in the base 'sam' model.
            #sam.load_state_dict(trained_state_dict, strict=False) # This works but might show warnings.
            # Let's manually filter the SAM keys for clarity.
            sam_state_dict_update = {k.replace('sam.', ''): v
                                    for k, v in trained_state_dict.items()
                                    if k.startswith('sam.') and 'image_encoder' not in k} # Only take keys that are in SAM and not image_encoder

            sam.load_state_dict(sam_state_dict_update, strict=False) # Load the filtered SAM keys


            # --- Instantiate and load the ClassificationHead (FIXED LOADING LOGIC) ---
            # num_classes is derived from your color map now
            classification_head = ClassificationHead(in_channels=1, num_classes=num_classes)
            classification_head.to(device=device) # Move classification head to device

            # Load the classification head's state_dict from the fine-tuned checkpoint
            # This is the section where we need to use the key re-mapping.
            cls_head_state_dict_numeric = {}
            for k, v in trained_state_dict.items():
                if k.startswith('classification_head.'):
                    # Store the value with the key AFTER removing the 'classification_head.' prefix
                    cls_head_state_dict_numeric[k.replace('classification_head.', '')] = v

            # Manually define the mapping from the saved numeric keys to the *NEW* expected named keys ('conv1', 'fc')
            # Based on the initial error's unexpected keys and the corrected ClassificationHead definition
            key_mapping = {
                "0.weight": "conv1.weight", # Map saved '0.' keys to instantiated 'conv1.'
                "0.bias": "conv1.bias",
                "4.weight": "fc.weight",    # Map saved '4.' keys to the NEW instantiated 'fc.'
                "4.bias": "fc.bias",
            }

            cls_head_state_dict_named = {}
            for saved_key_numeric, expected_key_named in key_mapping.items():
                if saved_key_numeric in cls_head_state_dict_numeric:
                    cls_head_state_dict_named[expected_key_named] = cls_head_state_dict_numeric[saved_key_numeric]
                else:
                    print(f"Warning: Expected classification head key '{saved_key_numeric}' not found in trained checkpoint for mapping.")

            # Load the re-mapped state dict into the classification head instance
            if cls_head_state_dict_named: # Only attempt to load if keys were found
                classification_head.load_state_dict(cls_head_state_dict_named, strict=True)
                print("Classification head weights loaded successfully.")
            else:
                 print("Warning: No classification head keys found in trained checkpoint. Classification will not work.")
                 classification_head = None # Set to None if not loaded

            classification_head.eval() if classification_head else None # Set to evaluation mode if loaded

        else:
             print(f"Warning: Fine-tuned checkpoint not found at '{args.sam_trained_checkpoint}'. Using base SAM without fine-tuned weights or classification.")
             classification_head = None # Classification head cannot be used without the trained checkpoint


        # --- Instantiate SAM Automatic Mask Generator ---
        # The Automatic Mask Generator uses the 'sam' model instance you just loaded.
        # It does NOT directly involve the ClassificationHead in its mask generation process.
        # It will use the fine-tuned prompt encoder and mask decoder if you loaded the trained_state_dict onto 'sam'.
        mask_generator = SamAutomaticMaskGenerator(
             model=sam,
             points_per_side=args.points_per_side,
             pred_iou_thresh=args.pred_iou_thresh,
             stability_score_thresh=args.stability_score_thresh,
             min_mask_region_area=args.min_mask_region_area,
             # Add other parameters if needed, see SamAutomaticMaskGenerator docs
             # box_nms_thresh=args.box_nms_thresh, # Example optional param
             # crop_n_layers=args.crop_n_layers,
             # crop_nms_thresh=args.crop_nms_thresh,
             # crop_overlap_ratio=args.crop_overlap_ratio,
             # crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
             # point_grids=args.point_grids,
             # box_threshold=args.box_threshold,
             # mask_threshold=args.mask_threshold,
        )
        print("Generating masks with SAM Automatic Mask Generator...")
        # Generate masks on the RGB image (after initial resize, before SAM internal transform)
        masks = mask_generator.generate(image_rgb)
        print(f"Generated {len(masks)} masks.")

    except Exception as e:
        print(f"Error during SAM or Mask Generation processing: {e}")
        # Print more detailed traceback for debugging
        import traceback
        traceback.print_exc()
        sys.exit(1) # Exit on error


    # --- Process Generated Masks for Classification ---
    # Now, iterate through the generated masks and use the classification_head to classify each one.
    # The classification head expects a Bx1x256x256 tensor of low-res masks.
    # SamAutomaticMaskGenerator gives you upsampled masks at the input image size (1892x1892 in this case).
    # We will resize the generated mask to 256x256 using nearest neighbor interpolation for classification input.
    # Note: This is an approximation, as the head was trained on predicted low-res *logits*, not resized masks.
    # If classification is poor, a more complex approach feeding prompts back through the mask decoder might be needed.

    classified_masks = []
    if classification_head:
        print("Classifying generated masks...")
        for i, mask_info in enumerate(masks):
            mask_seg_orig_size = mask_info['segmentation'] # Boolean mask at the initially resized image size (1892x1892)

            # Skip if mask is empty after generation
            if not np.any(mask_seg_orig_size):
                 print(f"Warning: Generated mask {i+1} is empty. Skipping classification and SVG.")
                 continue

            # Resize mask to 256x256 for classification head (using nearest neighbor for mask integrity)
            # Ensure input is float32 as expected by conv layer
            mask_resized_np = cv2.resize(mask_seg_orig_size.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
            mask_resized_tensor = torch.as_tensor(mask_resized_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # 1 x 1 x 256 x 256

            # Pass the resized mask through the classification head
            with torch.no_grad(): # No need for gradients during inference
                class_logits = classification_head(mask_resized_tensor) # 1 x num_classes

            # Get the predicted class index
            # Assume class 0 is background and we trained on classes 1-5, so argmax is sufficient.
            predicted_class_id = torch.argmax(class_logits, dim=1).squeeze().item() # scalar int

            # Store the predicted class ID with the mask info
            mask_info['predicted_class_id'] = predicted_class_id
            classified_masks.append(mask_info) # Add the mask info dict with the new class ID

        print(f"Classified {len(classified_masks)} masks.")
    else:
        # If classification_head wasn't loaded, just use the original masks without class info
        classified_masks = masks
        print("Classification head not loaded. Masks will not be colored by class.")
        # You might want to assign a default color or handle this in show_anns_with_contours

    # --- Ensure masks in classified_masks are boolean and at ORIGINAL image dimensions for SVG/Preview ---
    # SamAutomaticMaskGenerator returns masks at the *input* image size it processed (1892x1892 resized here).
    # show_anns_with_contours expects masks to match the `image` dimensions passed to it, which is original_image.
    # So, resize masks from 1892x1892 back up to original_width x original_height.
    final_classified_masks = []
    print(f"Resizing masks from SAM input size to original image size ({original_width}x{original_height})...")
    for mask_info in classified_masks:
         mask_seg_sam_size = mask_info['segmentation'] # Mask at SAM input size (1892x1892)

         if mask_seg_sam_size.shape[:2] != (original_height, original_width):
             mask_seg_original_size_np = cv2.resize(mask_seg_sam_size.astype(np.uint8), (original_width, original_height), interpolation=cv2.INTER_NEAREST)
             mask_info['segmentation'] = mask_seg_original_size_np.astype(bool)
         # else: mask is already correct size (shouldn't happen with initial 1892 resize)

         final_classified_masks.append(mask_info) # Add the updated mask info dict

    print(f"Final masks prepared at original size: {len(final_classified_masks)}")


    # --- Call the SVG generation function ---
    if args.output: output_svg_path = args.output
    else: output_svg_path = None; print("No output SVG path specified. Plot will be shown.")

    show_anns_with_contours(
        final_classified_masks, # Pass the list of masks with predicted class IDs
        original_image,         # Pass the original image for background/dimensions
        outline_color=args.outline,
        output_path=output_svg_path,
        smoothing_method=args.smoothing_method,
        smoothing_s_factor=args.smoothing_s,
        smoothing_sigma=args.smoothing_sigma,
        num_spline_points=args.spline_points,
        mask_alpha=args.alpha,
        rdp_epsilon=args.rdp_epsilon,
        class_colors_bgr=class_colors_bgr # Pass the color map
    )

if __name__ == "__main__":
    main()
