#!/home/chuck/venv/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import cv2 # or use skimage.io
from segment_anything import SamPredictor, sam_model_registry # Using SamPredictor for preprocessing helpers
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision.transforms import functional as F
import random
import warnings

# Suppress potential warnings from libraries
warnings.filterwarnings("ignore")

# --- Configuration ---
IMAGES_DIR = 'training_data/1. orthophotos/'
SEGMASKS_DIR = 'training_data/2. segmentation masks/' # Assumes these contain instance IDs
LABELMASKS_DIR = 'training_data/3. class masks/'     # Assumes these contain class IDs per pixel
CLASS_LABELS = {
    0: "Background",
    1: "Fairway",
    2: "Green",
    3: "Tee",
    4: "Bunker",
    5: "Water"
}
NUM_CLASSES = len(CLASS_LABELS)

# MODEL_TYPE = "vit_b" # Or "vit_l", "vit_h"
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = ".checkpoints/sam_vit_b_01ec64.pth" # Verify this path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Note "cpu' can litterally take days or weeks for this program.
# DEVICE="cpu"
print(f"Using device: {DEVICE}")

# Training Parameters
BATCH_SIZE = 2 # Adjust based on your GPU memory
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10 # Start with a small number
FREEZE_IMAGE_ENCODER = True # Keep SAM's large encoder frozen
# Use a combination of losses
SEGMENTATION_LOSS_WEIGHT = 1.0
CLASSIFICATION_LOSS_WEIGHT = 1.0

# SAM specific preprocessing setup
sam_model_for_predictor = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam_predictor = SamPredictor(sam_model_for_predictor)
sam_transform = ResizeLongestSide(sam_predictor.model.image_encoder.img_size)

# --- Dataset Class ---
class GolfCourseDataset(Dataset):
    def __init__(self, images_dir, segmasks_dir, labelmasks_dir, sam_transform):
        self.images_dir = images_dir
        self.segmasks_dir = segmasks_dir
        self.labelmasks_dir = labelmasks_dir
        self.sam_transform = sam_transform
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff'))])
        self.instances = []

        print("Indexing dataset instances...")
        # Iterate through images and find all instances (objects) with labels
        for img_name in self.image_files:
            base_name, _ = os.path.splitext(img_name)
            seg_mask_path = os.path.join(self.segmasks_dir, base_name + '.png') # Assuming mask file name matches image file name
            label_mask_path = os.path.join(self.labelmasks_dir, base_name + '.png') # Assuming label file name matches image file name
            image_path = os.path.join(self.images_dir, img_name)

            if not os.path.exists(seg_mask_path) or not os.path.exists(label_mask_path):
                print(f"Warning: Skipping {img_name} as mask files not found.")
                continue

            try:
                # Read masks - ensure they are read as is (might be 16-bit or 32-bit integers for IDs)
                seg_mask = cv2.imread(seg_mask_path, cv2.IMREAD_UNCHANGED)
                label_mask = cv2.imread(label_mask_path, cv2.IMREAD_UNCHANGED)

                if seg_mask is None or label_mask is None:
                     print(f"Warning: Could not read mask files for {img_name}.")
                     continue
                if seg_mask.ndim == 3:
                    # Assuming the relevant data is in the first channel, or convert to grayscale
                    # Option 1: Take the first channel (if you know the data is there)
                    # seg_mask = seg_mask[:, :, 0]
                    # Option 2: Convert to grayscale (safer if data might be spread or in BGR format)
                    if seg_mask.shape[2] == 3: # Check if it's a 3-channel image
                         seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2GRAY)
                    elif seg_mask.shape[2] == 4: # Check if it's a 4-channel image (like RGBA)
                         seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_BGRA2GRAY)
                    else:
                        print(f"Warning: seg_mask for {img_name} has unexpected number of channels ({seg_mask.shape[2]}). Attempting to use first channel.")
                        seg_mask = seg_mask[:, :, 0] # Fallback: just take the first channel

                # Also ensure label_mask is 2D if it's possibly 3D
                if label_mask.ndim == 3:
                     if label_mask.shape[2] == 3:
                         label_mask = cv2.cvtColor(label_mask, cv2.COLOR_BGR2GRAY)
                     elif label_mask.shape[2] == 4:
                         label_mask = cv2.cvtColor(label_mask, cv2.COLOR_BGRA2GRAY)
                     else:
                         print(f"Warning: label_mask for {img_name} has unexpected number of channels ({label_mask.shape[2]}). Attempting to use first channel.")
                         label_mask = label_mask[:, :, 0]


                if seg_mask.shape[:2] != label_mask.shape[:2]:
                    print(f"Warning: Mask shapes mismatch for {img_name}. Skipping.")
                    continue
                
                # Find unique instance IDs (excluding background 0)
                instance_ids = np.unique(seg_mask) # This should now work on a 2D array
                instance_ids = instance_ids[instance_ids != 0] # Ignore background ID
                 
                for instance_id in instance_ids:
                    # Create binary mask for this specific instance
                    instance_binary_mask = (seg_mask == instance_id).astype(np.uint8)

                    # Find the class label for this instance
                    # Take a pixel within the instance mask and get its label from the label mask
                    coords = np.argwhere(instance_binary_mask > 0)
                    if coords.shape[0] == 0: # Should not happen if instance_id came from unique, non-zero
                         print(f"Warning: No pixels found for instance {instance_id} in {img_name}. Skipping.")
                         continue
                    # Get the class label from the first pixel of the instance
                    first_pixel_coords = coords[0]
                    class_id = label_mask[first_pixel_coords[0], first_pixel_coords[1]]

                    # Ignore background class instances if they somehow exist in instance_ids
                    if class_id == 0:
                        continue

                    # Compute bounding box for the instance
                    y_coords, x_coords = np.where(instance_binary_mask > 0)
                    x_min, x_max = np.min(x_coords), np.max(x_coords)
                    y_min, y_max = np.min(y_coords), np.max(y_coords)
                    # Bounding box format [x_min, y_min, x_max, y_max]
                    bbox = [x_min, y_min, x_max, y_max]

                    self.instances.append({
                        'image_path': image_path,
                        'instance_binary_mask': instance_binary_mask, # Store the numpy array mask
                        'class_id': class_id,
                        'bbox': bbox # Store the numpy array bbox
                    })
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue

        print(f"Found {len(self.instances)} instances across {len(self.image_files)} images.")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance_info = self.instances[idx]

        # Load Image
        image = Image.open(instance_info['image_path']).convert('RGB')
        original_width, original_height = image.size
        original_size = (original_height, original_width) # SAM usually expects (H, W)
        image_np = np.array(image) # HWC, uint8

        # Get Mask and Class ID
        instance_binary_mask_np = instance_info['instance_binary_mask'] # HWC or HW, uint8 (binary)
        class_id = instance_info['class_id']
        bbox = instance_info['bbox'] # [x_min, y_min, x_max, y_max]

        # --- Apply SAM Transforms to Image and Bounding Box ---
        # SAM resizes the longest side to sam_predictor.model.image_encoder.img_size (e.g., 1024)
        # and pads the shorter side.
        transformed_image_np = self.sam_transform.apply_image(image_np) # HWC, uint8
        transformed_image = torch.as_tensor(transformed_image_np).permute(2, 0, 1).contiguous() # CHW, uint8

        # Transform Bounding Box
        # SAM transform expects boxes in XYXY format
        transformed_bbox_np = self.sam_transform.apply_boxes(np.array([bbox]), image_np.shape[:2]) # shape (1, 4)
        transformed_bbox = torch.as_tensor(transformed_bbox_np, dtype=torch.float) # shape (1, 4)


        # --- Resize Ground Truth Mask to SAM's Low Resolution Mask Output Size (256x256) ---
        # SAM's mask decoder outputs masks of size 256x256. We need to train against a mask of this size.
        # We use OpenCV for resizing the numpy mask array.
        target_mask_size = 256 # The spatial size of SAM's low_res_masks
        # Ensure the mask is 2D if it was read as 3D initially (should be fixed by previous step, but double check)
        if instance_binary_mask_np.ndim == 3:
             instance_binary_mask_np = instance_binary_mask_np[:, :, 0] # Or convert to grayscale

        # Resize mask using nearest neighbor interpolation
        resized_mask_np = cv2.resize(
            instance_binary_mask_np,
            (target_mask_size, target_mask_size), # Target (width, height)
            interpolation=cv2.INTER_NEAREST # Important for masks!
        ) # HW, uint8 (binary 0 or 1)

        # Convert the resized mask to a tensor, add channel dimension, and make it float
        # Add a channel dimension: HW -> CHW (1, H, W)
        resized_mask = torch.as_tensor(resized_mask_np).unsqueeze(0).float() # CHW (1, 256, 256), float


        # SAM's normalization (mean/std) is applied internally by the model's `preprocess` method
        # when using sam_predictor.model. We pass the raw transformed image.

        return {
            'image': transformed_image, # CHW, uint8
            'bbox': transformed_bbox,   # XYXY, float, shape (1, 4)
            'mask': resized_mask,       # CHW (1, 256, 256), float
            'class_label': torch.tensor(class_id, dtype=torch.long), # scalar, long
            'original_size': original_size      # (H, W) tuple
        }

# --- Model Definition ---
class SamFineTuner(nn.Module):
    def __init__(self, sam_checkpoint, model_type, num_classes, freeze_image_encoder=True):
        super().__init__()
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

        if freeze_image_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
            print("SAM Image Encoder frozen.")
        else:
             print("SAM Image Encoder NOT frozen.")

        # We will train the mask decoder and the prompt encoder (they are smaller)
        for param in self.sam.prompt_encoder.parameters():
             param.requires_grad = True
        for param in self.sam.mask_decoder.parameters():
             param.requires_grad = True


        # Add a classification head.
        # SAM's mask decoder outputs 'low_res_masks' which are 256x256 logits before upsampling.
        # We can pool these features and add a linear layer for classification.
        # The low_res_masks are B x 1 x 256 x 256 in the default forward pass.
        # The mask decoder also outputs 'iou_predictions' (B x 1) which are internal confidence scores.
        # A simple approach is to pool the low_res_masks or use features derived from them.
        # Let's pool the low_res_masks (output channel is 1 in default SAM)
        # Note: SAM's mask decoder structure is complex, accessing internal features is tricky.
        # A safer bet is processing the direct output 'low_res_masks'.
        self.classification_head = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # Add a small conv layer
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # Pool spatial dimensions to 1x1
            nn.Flatten(), # Flatten B x 32 x 1 x 1 to B x 32
            nn.Linear(32, num_classes) # Linear layer to output class logits
        )

    def forward(self, images, bboxes, original_sizes):
        # images: B x C x H x W (raw transformed image from dataset, uint8 or float)
        # bboxes: B x 1 x 4 (transformed box prompts from dataset, float)
        # original_sizes: List[(H, W)] for each image in the batch
        
        batched_input = []
        # The dataset yields batches where each item is an instance (image+box+mask+label).
        # So, the batch size B is the number of instances in this step.
        # We need to prepare a list of B dictionaries, one for each instance/image in the batch.
        print("Forward original_sizes = ", original_sizes)
        for i in range(images.shape[0]): # Iterate through batch dimension B
            print(f"Forward i = {i}")
            # Sam.forward expects the image already preprocessed
            input_image_i = self.sam.preprocess(images[i].float()) # Preprocess a single image (C, H', W')

            input_dict = {
                'image': input_image_i,
                # The docstring says 'boxes' shape Bx4 *for a single image dictionary*.
                # Since our dataset gives 1 box per instance, we need shape (1, 4) here.
                # Our input `bboxes` is B x 1 x 4. Slicing [i] gives 1 x 4, which is correct.
                'boxes': bboxes[i],
                # Note: Your Sam.forward definition also expects 'original_size'.
                # We might need to add this to the dataset and pass it here if SAM
                # uses it internally even when not upsampling. Let's try without first.
                # 'original_size': (original_height, original_width) # Need to get this from dataset
                'original_size': (int(original_sizes[0][i]), int(original_sizes[1][i])), # Convert to int tuple as expected
            }
            batched_input.append(input_dict)

        # Call the main sam.forward method with the structured input
        # The output is a List[Dict], where each dict corresponds to an input dict.
        # Each output dict contains 'masks', 'iou_predictions', and 'low_res_logits'.
        outputs: List[Dict[str, torch.Tensor]] = self.sam(batched_input, multimask_output=False)

        # Extract outputs from the list of dictionaries
        # We need low_res_logits for the segmentation loss and classification head.
        # low_res_logits in each output dict has shape (N, 1, 256, 256), where N is num prompts for that image.
        # In our case, N=1 for every image in the batch.
        low_res_masks_list = [output_dict['low_res_logits'] for output_dict in outputs]
        # Concatenate the list of tensors along the batch dimension (dim=0)
        low_res_masks = torch.cat(low_res_masks_list, dim=0) # Final shape B x 1 x 256 x 256

        # Pass low_res_masks through the classification head
        # low_res_masks shape is B x 1 x 256 x 256
        class_logits = self.classification_head(low_res_masks)

        # Return the predicted low-resolution mask logits and class logits
        return low_res_masks, class_logits


# Classification Loss: Cross Entropy for multi-class classification
segmentation_criterion = nn.BCEWithLogitsLoss()
classification_criterion = nn.CrossEntropyLoss()

# --- Training Function ---
def train_model(model, dataloader, optimizer, num_epochs, device, segmentation_criterion, classification_criterion):
    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        running_seg_loss = 0.0
        running_cls_loss = 0.0
        running_total_loss = 0.0

        for i, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            bboxes = batch['bbox'].to(device)
            true_masks = batch['mask'].to(device) # This is now the 256x256 mask
            class_labels = batch['class_label'].to(device)
            original_sizes = batch['original_size'] # Unpack the original sizes (list of tuples)
            optimizer.zero_grad()
            print(f"--- i = {i}")
            # Forward pass - model now returns low_res_masks_logits
            predicted_low_res_masks_logits, class_logits = model(images, bboxes, original_sizes) # Pass images, bboxes original_sizes

            # Calculate Segmentation Loss
            # Compare predicted_low_res_masks_logits (B, 1, 256, 256)
            # with true_masks (B, 1, 256, 256)
            # Ensure true_masks is float and same shape
            seg_loss = segmentation_criterion(predicted_low_res_masks_logits, true_masks)

            # Calculate Classification Loss
            # class_logits are B x NUM_CLASSES
            # class_labels are B
            cls_loss = classification_criterion(class_logits, class_labels)

            # Combined Loss
            total_loss = SEGMENTATION_LOSS_WEIGHT * seg_loss + CLASSIFICATION_LOSS_WEIGHT * cls_loss

            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()

            running_seg_loss += seg_loss.item()
            running_cls_loss += cls_loss.item()
            running_total_loss += total_loss.item()

            if (i + 1) % 10 == 0: # Print every 10 batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], "
                      f"Seg Loss: {running_seg_loss/10:.4f}, Cls Loss: {running_cls_loss/10:.4f}, "
                      f"Total Loss: {running_total_loss/10:.4f}")
                running_seg_loss = 0.0
                running_cls_loss = 0.0
                running_total_loss = 0.0

        print(f"Epoch [{epoch+1}/{num_epochs}] finished.")

        # Optional: Add evaluation on a validation set here
        # Optional: Save model checkpoint
        if (epoch + 1) % 5 == 0: # Save every 5 epochs
            save_path = f"./checkpoints/sam_finetuned_golf_epoch_{epoch+1}.pth"
            # Save the model's state_dict
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Setup Dataset and DataLoader
    golf_dataset = GolfCourseDataset(IMAGES_DIR, SEGMASKS_DIR, LABELMASKS_DIR, sam_transform)
    golf_dataloader = DataLoader(golf_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=30 ) # os.cpu_count() // 2) # Use half CPU cores

    # 2. Setup Model
    model = SamFineTuner(CHECKPOINT_PATH, MODEL_TYPE, NUM_CLASSES, freeze_image_encoder=FREEZE_IMAGE_ENCODER)
    model.to(DEVICE)

    # 3. Setup Optimizer
    # We only optimize parameters that require gradients (mask decoder, prompt encoder, classification head)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Start Training
    print("Starting training...")
    train_model(model, golf_dataloader, optimizer, NUM_EPOCHS, DEVICE, segmentation_criterion, classification_criterion)
    print("Training finished.")

    # --- How to use the fine-tuned model for inference ---
    # After training, the saved model_state_dict can be loaded.
    # The inference process would involve:
    # 1. Load the fine-tuned model state_dict into an instance of SamFineTuner.
    # 2. Load a new image, apply the same SAM preprocessing (ResizeLongestSide, then model.preprocess).
    # 3. Decide on a prompting strategy for inference (e.g., grid of points, predicted bounding boxes from another model, user clicks).
    # 4. Pass the preprocessed image and inference prompts through the model's forward pass.
    # 5. The model will output predicted mask logits and class logits for each prompt.
    # 6. Apply sigmoid to mask logits and threshold (e.g., >0.5) to get binary masks.
    # 7. Apply softmax to class logits to get class probabilities.
    # 8. Associate the predicted mask with the predicted class for each prompt.

    print("\nInference Usage Example (Conceptual):")
    print("Load the saved model:")
    print("model_inf = SamFineTuner(CHECKPOINT_PATH, MODEL_TYPE, NUM_CLASSES, freeze_image_encoder=True) # Match training config")
    print("model_inf.load_state_dict(torch.load('sam_finetuned_golf_epoch_XX.pth'))")
    print("model_inf.to(DEVICE)")
    print("model_inf.eval()")
    print("\nPrepare input image and prompts (e.g., predicted boxes or grid points):")
    print("# img_path = 'path/to/new/golf_image.jpg'")
    print("# image = Image.open(img_path).convert('RGB')")
    print("# image_np = np.array(image)")
    print("# transformed_image_np = sam_transform.apply_image(image_np)")
    print("# transformed_image = torch.as_tensor(transformed_image_np).permute(2, 0, 1).contiguous().to(DEVICE)")
    print("# inference_prompts = torch.tensor([[x_min, y_min, x_max, y_max], ...]).to(DEVICE) # Shape B x N x 4 or N x 4 if B=1")
    print("\nRun inference:")
    print("# with torch.no_grad():")
    print("#    predicted_masks_logits, class_logits = model_inf(transformed_image.unsqueeze(0), inference_prompts.unsqueeze(0)) # Add batch dim")
    print("\nProcess outputs:")
    print("# predicted_masks = torch.sigmoid(predicted_masks_logits) > 0.5")
    print("# class_probabilities = torch.softmax(class_logits, dim=-1)")
    print("# predicted_classes = torch.argmax(class_probabilities, dim=-1)")
    print("# Associate masks and classes for each prompt.")
