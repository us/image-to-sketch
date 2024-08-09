import base64
import subprocess
import cv2
import torch
import numpy as np
import streamlit as st
import supervision as sv
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from scipy.ndimage import gaussian_filter


# Streamlit settings
st.title('Segmentation Application')
st.write('Please upload an image.')

# Set parameters in the sidebar
st.sidebar.title("Parameters")
points_per_side = st.sidebar.slider("Points per Side", min_value=1, max_value=100, value=50)
pred_iou_thresh = st.sidebar.slider("Prediction IoU Threshold", min_value=0.0, max_value=1.0, value=0.70)
stability_score_thresh = st.sidebar.slider("Stability Score Threshold", min_value=0.0, max_value=1.0, value=0.85)
box_nms_thresh = st.sidebar.slider("Box NMS Threshold", min_value=0.0, max_value=1.0, value=0.90)
crop_n_layers = st.sidebar.slider("Crop Layers", min_value=0, max_value=10, value=0)
min_mask_region_area = st.sidebar.number_input("Minimum Mask Region Area", min_value=0, value=300)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()

# Load the SAM model (once)
@st.cache_resource
def load_model():
    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = "./weights/sam_vit_h_4b8939.pth"  # Update this path
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    return sam

sam = load_model()

# Image upload
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_rgb = np.array(image)

    # Resize image for faster processing
    image_rgb = cv2.resize(image_rgb, (512, 512))

    best_params = {
        "points_per_side": points_per_side,
        "pred_iou_thresh": pred_iou_thresh,
        "stability_score_thresh": stability_score_thresh,
        "box_nms_thresh": box_nms_thresh,
        "crop_n_layers": crop_n_layers,
        "min_mask_region_area": min_mask_region_area
    }

    @st.cache_data
    def generate_and_annotate(image, params):
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=params.get("points_per_side", 32),
            pred_iou_thresh=params.get("pred_iou_thresh", 0.88),
            stability_score_thresh=params.get("stability_score_thresh", 0.95),
            box_nms_thresh=params.get("box_nms_thresh", 0.7),
            crop_n_layers=params.get("crop_n_layers", 0),
            crop_n_points_downscale_factor=params.get("crop_n_points_downscale_factor", 1),
            min_mask_region_area=params.get("min_mask_region_area", 0)
        )
        masks = mask_generator.generate(image)
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        detections = sv.Detections.from_sam(sam_result=masks)
        annotated_image = mask_annotator.annotate(scene=np.ones_like(image) * 255, detections=detections)
        return annotated_image, masks

    @st.cache_data
    def draw_segmentation_outlines(masks, image_shape):
        outline_image = np.ones((image_shape[0], image_shape[1], 3), dtype=np.uint8) * 255  # White background
        for mask_dict in masks:
            mask = mask_dict.get('segmentation')  # Extract the mask array from the dictionary
            if mask is not None:
                mask = np.array(mask).astype(np.uint8)
                mask_resized = cv2.resize(mask, (image_shape[1], image_shape[0]))  # Resize mask
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(outline_image, contours, -1, (0, 0, 0), 5, lineType=cv2.LINE_AA)  # Draw contours in black
        return outline_image

    
    @st.cache_data
    def zhang_suen_thinning(image):
        def neighbours(x, y, image):
            "Return 8-neighbours of image point P1(x,y), in a clockwise order"
            img = image
            return [img[x-1, y], img[x-1, y+1], img[x, y+1], img[x+1, y+1], img[x+1, y], img[x+1, y-1], img[x, y-1], img[x-1, y-1]]

        def transitions(neighbours):
            "No. of 0,1 patterns (0=white, 1=black) in the ordered sequence"
            n = neighbours + neighbours[0:1]    # P2, P3, P4, P5, P6, P7, P8, P9, P2
            return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

        def zhang_suen_iteration(image, iter_num):
            changing1 = changing2 = 1
            img = image.copy()
            while changing1 or changing2:   # iterates until no further changes occur in both phases
                changing1 = []
                for x in range(1, img.shape[0] - 1):
                    for y in range(1, img.shape[1] - 1):
                        P2, P3, P4, P5, P6, P7, P8, P9 = neighbours(x, y, img)
                        if (img[x, y] == 1 and  # black pixel
                                2 <= sum(neighbours(x, y, img)) <= 6 and  # black pixel is surrounded by 2 to 6 black pixels
                                transitions(neighbours(x, y, img)) == 1 and  # black pixel is surrounded by exactly one 0-1 transition
                                P2 * P4 * P6 == 0 and  # at least one of P2, P4, P6 is white
                                P4 * P6 * P8 == 0):  # at least one of P4, P6, P8 is white
                            changing1.append((x, y))
                for x, y in changing1:
                    img[x, y] = 0
                changing2 = []
                for x in range(1, img.shape[0] - 1):
                    for y in range(1, img.shape[1] - 1):
                        P2, P3, P4, P5, P6, P7, P8, P9 = neighbours(x, y, img)
                        if (img[x, y] == 1 and  # black pixel
                                2 <= sum(neighbours(x, y, img)) <= 6 and  # black pixel is surrounded by 2 to 6 black pixels
                                transitions(neighbours(x, y, img)) == 1 and  # black pixel is surrounded by exactly one 0-1 transition
                                P2 * P4 * P8 == 0 and  # at least one of P2, P4, P8 is white
                                P2 * P6 * P8 == 0):  # at least one of P2, P6, P8 is white
                            changing2.append((x, y))
                for x, y in changing2:
                    img[x, y] = 0
            return img

        image = image.copy() // 255  # convert to binary (0, 1)
        thinned_image = zhang_suen_iteration(image, 0)
        return thinned_image * 255  # convert back to 0-255
    
    @st.cache_data
    def redetect_and_draw_thin_lines(image, thickness=1):
        # Ensure the image is in grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        img = cv2.bitwise_not(image)

        # Apply the Zhang-Suen thinning algorithm
        skeleton = zhang_suen_thinning(img)

        # Apply morphological closing to smooth the lines
        kernel_size = max(1, thickness)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel)

        # Smooth the thinned lines
        skeleton = cv2.GaussianBlur(skeleton, (kernel_size, kernel_size), 0)

        # Ensure the background is white and the lines are black
        skeleton = cv2.bitwise_not(skeleton)
        
        return skeleton

    # Example usage:
    # thin_line_image = redetect_and_draw_thin_lines(contour_image, thickness=line_thickness)
    # Function to adjust contrast
    # Function to convert binary image to grayscale
    def binary_to_grayscale(binary_image):
        grayscale_image = binary_image.astype(np.uint8) * 255
        return grayscale_image

    # Function to apply Gaussian Blur
    def apply_gaussian_blur(image, sigma=1.5):
        blurred_image = gaussian_filter(image, sigma=sigma)
        return blurred_image 

    from skimage import exposure
    def adjust_contrast(image, gamma=20.0):
        adjusted_image = exposure.adjust_gamma(image, gamma)
        return adjusted_image
    
    @st.cache_data
    def apply_image_enhancements(image):
        # Apply the process to the previously loaded binary image
        binary_image = image > 128  # Assume binary threshold at the middle
        grayscale_image = binary_to_grayscale(binary_image)
        blurred_image = apply_gaussian_blur(grayscale_image, sigma=1.5)
        adjusted_image = adjust_contrast(blurred_image, gamma=20)
        return adjusted_image


    def convert_png_to_bmp(png_file_path):
        with Image.open(png_file_path) as img:
            bmp_file_path = png_file_path.rsplit('.', 1)[0] + '.bmp'
            img.save(bmp_file_path)
            return bmp_file_path

    def convert_bmp_to_svg(bmp_file_path, svg_file_path):
        subprocess.run(['potrace', '-s', bmp_file_path, '-o', svg_file_path], check=True)

    def convert_png_to_svg(png_file_path, svg_file_path):
        bmp_file_path = convert_png_to_bmp(png_file_path)
        convert_bmp_to_svg(bmp_file_path, svg_file_path)

    def render_svg(svg):
        """Renders the given svg string."""
        b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
        html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
        st.write(html, unsafe_allow_html=True)
        

    # Generate masks and annotate image (run once)
    annotated_image, masks = generate_and_annotate(image_rgb, best_params)

    # # Display the results
    st.image(annotated_image, caption='Segmented Image', use_column_width=True)

      # Draw segmentation outlines and display them
    contour_image = draw_segmentation_outlines(masks, (955, 921, 3))
    st.image(contour_image, caption='Segmentation Outlines on White Background (3px)', use_column_width=True)

    # Redetect lines and draw thin lines on a new image
    thin_line_image = redetect_and_draw_thin_lines(contour_image)
    st.image(thin_line_image, caption='Redetected Segmentation Outlines on White Background (1px)', use_column_width=True)

    # Apply image enhancements    
    enhanced_image = apply_image_enhancements(thin_line_image)
    st.image(enhanced_image, caption='Enhanced Image', use_column_width=True)

    enhanced_image_path = 'enhanced_image.png'
    cv2.imwrite(enhanced_image_path, enhanced_image)

    svg_file_path = 'output.svg'
    # Automatically convert to SVG after image processing
    convert_png_to_svg(enhanced_image_path, svg_file_path)

    # Read and display the SVG
    with open(svg_file_path, 'r') as file:
        svg_data = file.read()
    render_svg(svg_data)
    st.success(f'SVG successfully saved as {svg_file_path}')



    # Generate masks and annotate image (run once)
    annotated_image, masks = generate_and_annotate(image_rgb, best_params)

    #   # Draw segmentation outlines and display them
    # contour_image = draw_segmentation_outlines(masks, image_rgb.shape)

    # # Redetect lines and draw thin lines on a new image
    # thin_line_image = redetect_and_draw_thin_lines(contour_image)

    # # Display the results
    # st.image(annotated_image, caption='Segmented Image', use_column_width=True)
    # st.image(contour_image, caption='Segmentation Outlines on White Background (3px)', use_column_width=True)
    # st.image(thin_line_image, caption='Redetected Segmentation Outlines on White Background (1px)', use_column_width=True)
