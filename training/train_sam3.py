import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

import numpy as np
import cv2 # OpenCV is needed for connected components and image loading
import os
from tqdm import tqdm
import matplotlib.pyplot as plt # For visualization/debugging (optional)
import random
import time # For timestamp in save path potentially
import torch.multiprocessing # For setting start method

# --- Configuration ---
# TODO: Adjust these paths before running
IMAGES_DIR = 'training_data/1. orthophotos/'
SEGMASKS_DIR = 'training_data/2. segmentation masks/'

# Choose model type and corresponding checkpoint
MODEL_TYPE = "vit_b" # Or "vit_l", "vit_h"
# Download from https://github.com/facebookresearch/segment-anything#model-checkpoints
# TODO: Adjust path to your downloaded checkpoint
CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters (adjust as needed)
LEARNING_RATE = 1e-5 # Fine-tuning often uses small learning rates
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 20 # Start with a reasonable number, increase if needed
BATCH_SIZE = 2 # Adjust based on GPU memory (set to 1 if memory is tight)
NUM_WORKERS = 2 # Number of parallel workers for data loading (set to 0 if issues persist)
# TODO: Adjust minimum instance size (in pixels) to filter noise
MIN_INSTANCE_AREA = 50
# Where to save the fine-tuned decoder weights
timestamp = time.strftime("%Y%m%d-%H%M%S")
SAVE_PATH = f"finetuned_sam_{MODEL_TYPE}_golf_decoder_{timestamp}.pth"
SAVE_EPOCH_INTERVAL = 5 # How often to save intermediate checkpoints

# Class labels (for reference and iterating)
CLASS_LABELS = {
    0: "Background",
    1: "Fairway",
    2: "Green",
    3: "Tee",
    4: "Bunker",
    5: "Water"
}
# Classes to generate prompts for (excluding background)
TARGET_CLASSES = [1, 2, 3, 4, 5]

# --- Helper Function for Instance-Based Prompt Generation ---
def get_instance_bounding_boxes(mask, class_id, min_area_threshold=10):
    """
    Gets bounding boxes and label indices for each instance of a given class_id
    in the mask using connected components.

    Args:
        mask (np.ndarray): The segmentation mask (H, W) with integer class labels.
        class_id (int): The target class ID to find instances for.
        min_area_threshold (int): Minimum pixel area for an instance to be considered.

    Returns:
        tuple: A tuple containing:
            - list: A list of tuples, where each tuple is (bounding_box, label_index).
                    bounding_box is a numpy array [x_min, y_min, x_max, y_max].
                    label_index is the unique integer label for this component in labels_im.
            - np.ndarray or None: The labeled image (H, W) where each pixel of the
                                   target class has a unique instance label (>=1),
                                   or None if no components are found.
    """
    boxes_and_labels = []
    # Create a binary mask for the specific class_id
    binary_mask = (mask == class_id).astype(np.uint8)

    # Find connected components
    # connectivity=8 considers pixels touching diagonally as connected
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    if num_labels <= 1: # Only background component (label 0) found
        return [], None # No instances of this class_id found

    # Iterate through components found (label 0 is background, skip it)
    for label_idx in range(1, num_labels):
        stat = stats[label_idx]
        area = stat[cv2.CC_STAT_AREA]

        # Filter small components based on area
        if area >= min_area_threshold:
            x = stat[cv2.CC_STAT_LEFT]
            y = stat[cv2.CC_STAT_TOP]
            w = stat[cv2.CC_STAT_WIDTH]
            h = stat[cv2.CC_STAT_HEIGHT]

            # Ensure width and height are positive
            if w > 0 and h > 0:
                # Create bounding box in xyxy format
                bbox = np.array([x, y, x + w, y + h])
                boxes_and_labels.append((bbox, label_idx))

    # Return the list of (box, label_index) tuples and the image labeled with instance indices
    return boxes_and_labels, labels_im

# --- Loss Functions ---
class DiceLoss(nn.Module):
    """Calculates Dice Loss."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predicted_probs, target_mask):
        # Flatten prediction and target tensors
        predicted_probs = predicted_probs.contiguous().view(-1)
        target_mask = target_mask.contiguous().view(-1)

        intersection = (predicted_probs * target_mask).sum()
        dice_coeff = (2. * intersection + self.smooth) / (predicted_probs.sum() + target_mask.sum() + self.smooth)
        return 1. - dice_coeff

class CombinedLoss(nn.Module):
    """Combines Dice Loss and BCEWithLogitsLoss."""
    def __init__(self, weight_dice=0.8, weight_bce=0.2):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss() # Numerically stable loss for logits
        self.w_dice = weight_dice
        self.w_bce = weight_bce

    def forward(self, predicted_logits, target_mask):
        # Target mask should be binary (0 or 1) and float for loss calculation
        target_mask_float = target_mask.float()
        # Calculate BCE loss on logits
        loss_bce = self.bce_loss(predicted_logits, target_mask_float)
        # Calculate Dice loss on probabilities (apply sigmoid)
        loss_dice = self.dice_loss(predicted_logits.sigmoid(), target_mask_float)
        # Combine losses
        combined_loss = self.w_dice * loss_dice + self.w_bce * loss_bce
        return combined_loss


# --- Custom Dataset ---
class GolfDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform # SAM's ResizeLongestSide transformer
        # Assumes corresponding mask has same base name but potentially different extension
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"Found {len(self.image_filenames)} images in {image_dir}.")

        # Precompute SAM's normalization constants on the correct device
        # These are standard for models trained on ImageNet
        self.register_buffer('pixel_mean', torch.tensor([123.675, 116.28, 103.53], device=DEVICE).view(-1, 1, 1), persistent=False)
        self.register_buffer('pixel_std', torch.tensor([58.395, 57.12, 57.375], device=DEVICE).view(-1, 1, 1), persistent=False)
        # Using register_buffer ensures they are moved to the correct device with the model/dataset
        # and are not considered model parameters. Set persistent=False if they don't need saving.

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Construct potential mask filenames (try common extensions)
        base_name = os.path.splitext(img_name)[0]
        possible_mask_names = [f"{base_name}.png", f"{base_name}.jpg", f"{base_name}.jpeg", f"{base_name}.tif"]
        mask_path = None
        for mask_name_try in possible_mask_names:
            potential_path = os.path.join(self.mask_dir, mask_name_try)
            if os.path.exists(potential_path):
                mask_path = potential_path
                break

        if mask_path is None:
            raise FileNotFoundError(f"Could not find mask for image {img_name} in {self.mask_dir} with common extensions.")

        # Load image (using OpenCV) - reads as uint8 BGR
        image = cv2.imread(img_path)
        if image is None:
             raise IOError(f"Could not read image: {img_path}")
        print(f"DEBUG 1: Original image shape for {img_name}: {image.shape}") # DEBUG
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to uint8 RGB

        # Load segmentation mask (ensure it's loaded as grayscale for class labels)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Could not read mask: {mask_path}")

        # --- Preprocessing Image ---
        # Resize image using SAM's transformer (maintains uint8)
        try:
            resized_image = self.transform.apply_image(image) # Output should be (1024, 1024, 3) uint8
            print(f"DEBUG 2: Resized image shape for {img_name}: {resized_image.shape}") # DEBUG
        except Exception as e:
            print(f"ERROR during apply_image for {img_name}: {e}")
            raise e

        # Convert numpy array to tensor (preserves uint8 initially)
        image_tensor = torch.as_tensor(resized_image, device=DEVICE)
        # PyTorch uses (C, H, W) format
        image_tensor = image_tensor.permute(2, 0, 1).contiguous()

        # Convert to float and Normalize (required by the model)
        image_tensor = image_tensor.float() # Convert to float32
        image_tensor = (image_tensor - self.pixel_mean) / self.pixel_std

        # --- Preprocessing Mask ---
        target_size = self.transform.target_length # Should be 1024
        # Resize mask using nearest neighbor interpolation to preserve class labels
        # Must match the spatial dimensions of the resized image tensor (1024x1024)
        resized_mask_cv = cv2.resize(mask, (target_size, target_size),
                                     interpolation=cv2.INTER_NEAREST)
        # Convert mask to LongTensor (required for integer class labels)
        mask_tensor = torch.as_tensor(resized_mask_cv, dtype=torch.long, device=DEVICE)

        # Final checks before returning
        print(f"DEBUG 3: Final image tensor shape in __getitem__: {image_tensor.shape}") # DEBUG
        print(f"DEBUG 4: Final image tensor dtype in __getitem__: {image_tensor.dtype}") # DEBUG

        # Return original image size tuple (height, width)
        original_image_size = tuple(image.shape[:2])

        return image_tensor, mask_tensor, original_image_size

# --- Main Training Function ---
def main():
    print(f"Using device: {DEVICE}")
    print("Loading SAM model...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam.train() # Set model to training mode (affects dropout, batchnorm etc.)

    # --- DEBUG PRINTS for Model Config ---
    print(f"DEBUG 5: SAM expected input size (sam.image_encoder.img_size): {sam.image_encoder.img_size}") # DEBUG
    if hasattr(sam.image_encoder, 'pos_embed') and sam.image_encoder.pos_embed is not None:
         print(f"DEBUG 6: Loaded Positional embedding shape: {sam.image_encoder.pos_embed.shape}") # DEBUG
    else:
         print("DEBUG: No 'pos_embed' attribute found directly on image_encoder.")
    # --- End DEBUG PRINTS ---

    # Freeze image encoder (transfer learning)
    print("Freezing image encoder parameters...")
    for param in sam.image_encoder.parameters():
        param.requires_grad = False
    # Optional: Freeze prompt encoder if you only want to train the decoder
    # print("Freezing prompt encoder parameters...")
    # for param in sam.prompt_encoder.parameters():
    # 	param.requires_grad = False

    # Parameters to optimize: only mask decoder
    # (and prompt encoder if not frozen above)
    params_to_optimize = list(sam.mask_decoder.parameters())
    # if not prompt_encoder_frozen: # Add logic based on freezing decision
    #     params_to_optimize += list(sam.prompt_encoder.parameters())

    optimizer = optim.AdamW(params_to_optimize,
                           lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)

    # Loss function
    criterion = CombinedLoss()

    # Prepare dataset and dataloader
    # Create the transformer using the model's expected input size
    sam_transform = ResizeLongestSide(sam.image_encoder.img_size)
    dataset = GolfDataset(IMAGES_DIR, SEGMASKS_DIR, transform=sam_transform)

    dataloader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True, # Shuffle data each epoch
                            num_workers=NUM_WORKERS,
                            pin_memory=True if DEVICE.type == 'cuda' else False) # Helps speed up CPU->GPU transfer

    print(f"Dataset size: {len(dataset)}, DataLoader size: {len(dataloader)}")
    print("Starting training...")

    # Variables for tracking loss
    total_loss_per_epoch = []

    for epoch in range(NUM_EPOCHS):
        epoch_loss_sum = 0.0
        num_prompts_processed_in_epoch = 0
        sam.train() # Ensure model is in train mode each epoch

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for batch_idx, batch_data in enumerate(pbar):
            if batch_data is None:
                print(f"Warning: Received None batch data at index {batch_idx}. Skipping.")
                continue
            try:
                 batch_images, batch_masks, batch_orig_sizes = batch_data
            except Exception as e:
                print(f"Error unpacking batch data at index {batch_idx}: {e}")
                continue # Skip this batch


            batch_loss_accumulator = 0.0 # Accumulates loss for the entire batch
            valid_items_in_batch = 0 # Counts items that contributed to loss

            # Process each item in the batch individually (simplifies prompt handling)
            # Note: BATCH_SIZE > 1 requires careful handling if batching prompts/embeddings
            for i in range(batch_images.shape[0]):
                image_tensor = batch_images[i]
                mask_tensor_cpu_numpy = batch_masks[i].cpu().numpy()
                # original_size = batch_orig_sizes[i] # Not directly used here

                # --- 1. Get Image Embeddings ---
                with torch.no_grad(): # No grad for frozen encoder
                    # Add batch dimension for the encoder
                    input_image_torch = image_tensor.unsqueeze(0)

                    # --- DEBUG PRINTS for Encoder Input ---
                    print(f"DEBUG 7 Batch {batch_idx}, Item {i}: Input tensor shape to encoder: {input_image_torch.shape}") # DEBUG
                    print(f"DEBUG 8 Batch {batch_idx}, Item {i}: Input tensor dtype to encoder: {input_image_torch.dtype}") # DEBUG
                    # --- End DEBUG PRINTS ---

                    try:
                        # >>> ERROR LIKELY HERE OR INSIDE if shapes mismatch <<<
                        image_embedding = sam.image_encoder(input_image_torch)
                    except RuntimeError as e:
                        print(f"\n!!! RUNTIME ERROR during image_encoder forward pass !!!")
                        print(f"Batch Index: {batch_idx}, Item Index in Batch: {i}")
                        print(f"Input Shape: {input_image_torch.shape}")
                        if hasattr(sam.image_encoder, 'pos_embed') and sam.image_encoder.pos_embed is not None:
                             print(f"Positional Embedding Shape: {sam.image_encoder.pos_embed.shape}")
                        print(f"Error: {e}")
                        # Depending on the error, you might want to investigate the specific image/mask
                        # img_name = dataset.image_filenames[dataloader.batch_sampler.sampler.first_index + batch_idx * BATCH_SIZE + i] # Get filename (approximate if shuffled)
                        # print(f"Potential problematic image index (approx if shuffled): {dataloader.batch_sampler.sampler.first_index + batch_idx * BATCH_SIZE + i}")
                        raise e # Re-raise the error to stop execution

                # --- 2. Generate INSTANCE Prompts and Train on them ---
                item_loss_sum = 0.0 # Sum of losses for all prompts for this single image
                prompts_for_item = 0 # Count of prompts processed for this image

                # Iterate through target classes (fairway, green, etc.)
                for class_id in TARGET_CLASSES:
                    # Get bounding boxes for ALL instances of this class_id
                    instance_boxes_labels, labels_im = get_instance_bounding_boxes(
                        mask_tensor_cpu_numpy, class_id, min_area_threshold=MIN_INSTANCE_AREA
                    )

                    # Skip if no instances of this class found, or if labels_im is None
                    if not instance_boxes_labels or labels_im is None:
                        continue

                    # Convert the labels_im (instance map) to a tensor ONCE per class
                    labels_im_tensor = torch.as_tensor(labels_im, device=DEVICE)

                    # Iterate through each detected INSTANCE box for this class
                    for bbox_coords, current_label_idx in instance_boxes_labels:
                        # Transform box numpy array [xmin, ymin, xmax, ymax] to tensor [1, 4]
                        box_torch = torch.as_tensor(bbox_coords, dtype=torch.float, device=DEVICE).unsqueeze(0)

                        # --- 3. Predict Masks with INSTANCE Prompt ---
                        # Ensure gradients are enabled for trainable parts (decoder)
                        with torch.set_grad_enabled(True):
                            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                                points=None, # Not using point prompts here
                                boxes=box_torch,
                                masks=None, # Not using mask prompts here
                            )

                            # Predict masks
                            low_res_masks, iou_predictions = sam.mask_decoder(
                                image_embeddings=image_embedding.detach(), # Detach from frozen encoder graph
                                image_pe=sam.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                multimask_output=False, # Train on single "best" mask output
                            )

                            # Upscale mask to model input size (e.g., 1024x1024)
                            upscaled_masks = sam.postprocess_masks(
                                low_res_masks,
                                input_size=tuple(image_tensor.shape[-2:]), # (1024, 1024)
                                original_image_size=tuple(image_tensor.shape[-2:]) # Match GT mask size
                            ) # Shape: (B=1, 1, H, W) e.g., (1, 1, 1024, 1024)

                        # --- 4. Calculate Loss for this specific INSTANCE ---
                        # Create the binary ground truth mask for THIS SPECIFIC INSTANCE
                        gt_instance_mask = (labels_im_tensor == current_label_idx).unsqueeze(0).unsqueeze(0) # Shape: (1, 1, H, W)

                        # Calculate loss
                        loss = criterion(upscaled_masks, gt_instance_mask.float())

                        # Check for NaN/Inf loss
                        if torch.isnan(loss) or torch.isinf(loss):
                           print(f"Warning: NaN/Inf loss encountered for class {class_id}, instance label {current_label_idx}. Skipping loss contribution.")
                           continue # Skip this prompt's loss

                        item_loss_sum += loss
                        prompts_for_item += 1
                        # --- End of loop for instance boxes ---
                    # --- End of loop for classes ---

                # --- 5. Average loss for the image and accumulate for batch ---
                if prompts_for_item > 0:
                    average_item_loss = item_loss_sum / prompts_for_item
                    batch_loss_accumulator += average_item_loss # Add the image's average loss to batch total
                    valid_items_in_batch += 1
                # --- End of loop for items in batch ---

            # --- Optimizer Step (after processing the batch) ---
            if valid_items_in_batch > 0:
                 # Average the accumulated loss over the number of valid items processed in the batch
                 effective_batch_loss = batch_loss_accumulator / valid_items_in_batch

                 optimizer.zero_grad()
                 effective_batch_loss.backward()
                 # Optional: Gradient Clipping (prevents exploding gradients)
                 # torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0)
                 optimizer.step()

                 # Accumulate loss sum for epoch average calculation
                 # Multiply by valid_items_in_batch because effective_batch_loss is already averaged
                 epoch_loss_sum += effective_batch_loss.item() * valid_items_in_batch
                 # Accumulate total number of prompts processed in the epoch
                 # Note: Need to sum prompts_for_item across the inner loop if BATCH_SIZE > 1.
                 # Simplified: Assume prompts_for_item holds count for last item if BS=1, need proper sum if BS > 1.
                 # Let's recalculate prompts processed per batch for accuracy:
                 prompts_in_batch = sum(prompts_for_item for item_idx in range(valid_items_in_batch)) # Need correct logic here
                 # num_prompts_processed_in_epoch += prompts_in_batch # TODO: Fix prompt counting across batch items

                 # Update progress bar with the loss for the current batch
                 pbar.set_postfix({"Loss": effective_batch_loss.item()})
            else:
                pbar.set_postfix({"Loss": 0}) # No valid items/prompts in batch

        # --- End of Epoch ---
        # Calculate average loss for the epoch (average loss per image)
        avg_epoch_loss = epoch_loss_sum / len(dataset) if len(dataset) > 0 else 0
        total_loss_per_epoch.append(avg_epoch_loss)
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} finished. Average Loss per Image: {avg_epoch_loss:.6f}")

        # --- Save model checkpoint periodically or at the end ---
        if (epoch + 1) % SAVE_EPOCH_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS:
             save_checkpoint_path = SAVE_PATH.replace(".pth", f"_epoch{epoch+1}.pth")
             # Save only the mask decoder state dict (most common fine-tuning scenario)
             torch.save(sam.mask_decoder.state_dict(), save_checkpoint_path)
             # If you also trained the prompt encoder, save it too:
             # torch.save(sam.prompt_encoder.state_dict(), save_checkpoint_path.replace("decoder", "prompt_encoder"))
             print(f"Saved fine-tuned mask decoder to {save_checkpoint_path}")

    print("Training complete.")
    # Final save (optional, could just use last periodic save)
    torch.save(sam.mask_decoder.state_dict(), SAVE_PATH)
    print(f"Final fine-tuned mask decoder saved to {SAVE_PATH}")

    # Optional: Plot training loss
    # plt.figure()
    # plt.plot(range(1, NUM_EPOCHS + 1), total_loss_per_epoch, marker='o')
    # plt.title("Training Loss per Epoch")
    # plt.xlabel("Epoch")
    # plt.ylabel("Average Loss per Image")
    # plt.grid(True)
    # plt.savefig(f"training_loss_curve_{timestamp}.png")
    # print(f"Saved training loss curve to training_loss_curve_{timestamp}.png")


if __name__ == "__main__":
    # Set the start method for multiprocessing with CUDA BEFORE any CUDA calls or DataLoader init
    # This is crucial for 'spawn' method required by CUDA on Linux/macOS.
    try:
        current_context = torch.multiprocessing.get_context()
        if current_context.get_start_method() != 'spawn':
             torch.multiprocessing.set_start_method('spawn', force=True)
             print("Set multiprocessing start method to 'spawn'.")
        else:
             print("Multiprocessing start method already 'spawn'.")
    except RuntimeError as e:
        # Might happen if context is already set and 'force=True' isn't enough (rare)
        print(f"Warning: Could not set start method ('spawn'): {e}")
        print("Ensure this script is run directly and not imported where context might be pre-set.")

    # Call the main training function
    main()
