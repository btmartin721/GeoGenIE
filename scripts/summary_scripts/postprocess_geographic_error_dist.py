import cv2
import numpy as np
from PIL import Image

# Load the image
image_path = "/Users/btm002/Documents/wtd/GeoGenIE/analyses/all_model_outputs_final_best_model_boot_5May2024/plots/emb_none_meth_kmeans_tr_0.8_vr_.10_lrf_0.75_fac_0.9_wt_none_bins_5_ndist_1000_scl_100_km_true_kd_false_out_true_bs_64_wp_1.0_crit_rmse_mac_3_geographic_error_test.png"

img = Image.open(image_path)

# Convert to numpy array for processing
img_array = np.array(img)


# Function to crop the image
def crop_image(image, left, top, right, bottom):
    return image[top:bottom, left:right]


img_pil = Image.fromarray(img_array)


# Update with coordinates based on visual inspection
crop_left, crop_top, crop_right, crop_bottom = (
    10,
    0,
    2271,
    2271,
)

img_array = crop_image(
    img_array, left=crop_left, top=crop_top, right=crop_right, bottom=crop_bottom
)

outpath = "/Users/btm002/Documents/wtd/GeoGenIE/analyses/all_model_outputs_final_best_model_boot_5May2024/plots/emb_none_meth_kmeans_tr_0.8_vr_.10_lrf_0.75_fac_0.9_wt_none_bins_5_ndist_1000_scl_100_km_true_kd_false_out_true_bs_64_wp_1.0_crit_rmse_mac_3_geographic_error_test_postprocessed.png"

img_pil = Image.fromarray(img_array)

# Save the result
img_pil.save(outpath)

img = cv2.imread(image_path)

# Check if the image was read successfully
if img is None:
    raise ValueError(f"Error loading image at {image_path}")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a binary threshold to isolate the shadowbox of the legend
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the contour resembling the shadowbox
legend_contour = None
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    # Adjust these conditions based on your image's shadowbox size and aspect ratio
    if 1.5 < aspect_ratio < 5.0 and 30 < w < 200 and 10 < h < 100:
        legend_contour = contour
        break

if legend_contour is not None:
    # Create a mask for the legend shadowbox
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [legend_contour], -1, (255, 255, 255), -1)

    # Apply the mask to the image to remove the region inside the shadowbox
    img[mask == 255] = 255
else:
    print("Legend shadowbox not found.")


# Convert back to PIL to save the result
img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

img_pil.save(outpath)
