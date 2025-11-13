"""
AI Image Cropper - Gradio UI
Interactive web interface for the AI image cropping tool with AI detection.
Author: Gary Stafford
License: MIT
"""

# Standard library imports
import base64
import logging
import traceback
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

# Third-party imports
import cv2
import gradio as gr
import numpy as np
from PIL import Image

# Local imports
from cropper import (
    DETR_AVAILABLE,
    RTDETR_AVAILABLE,
    ULTRALYTICS_AVAILABLE,
    ImageCropper,
)
from config import (
    BATCH_IMAGE_QUALITY,
    BATCH_OUTPUT_DIR,
    CONFIDENCE_MAX,
    CONFIDENCE_MIN,
    CONFIDENCE_STEP,
    CROP_LOGO_PATH,
    DEFAULT_ASPECT_RATIO_PRECISION,
    DEFAULT_CONFIDENCE,
    DEFAULT_PADDING,
    DEFAULT_THRESHOLD,
    INFO_SEPARATOR_WIDTH,
    LOGO_SIZE,
    PADDING_MAX,
    PADDING_MIN,
    PADDING_STEP,
    SAMPLE_IMAGE,
    SERVER_HOST,
    SERVER_PORT,
    SHARE_PUBLICLY,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    THRESHOLD_STEP,
    UI_IMAGE_HEIGHT,
    VALID_IMAGE_EXTENSIONS,
    YOLO_MODEL_PATH,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectionMethod(Enum):
    """Enumeration of available detection methods."""

    CONTOUR = "contour"
    SALIENCY = "saliency"
    EDGE = "edge"
    GRABCUT = "grabcut"
    DETR = "detr"
    RTDETR = "rt-detr"
    YOLO = "yolo"


class AspectMode(Enum):
    """Enumeration of aspect ratio modes."""

    NONE = "none"
    ORIGINAL = "original"
    CUSTOM = "custom"


class ProcessingResult(NamedTuple):
    """Result of image processing operation."""

    visualization: Optional[np.ndarray]
    cropped_image: Optional[Image.Image]
    info_text: str
    detections: Optional[List[Dict[str, Union[str, float, List[int]]]]]
    dropdown_update: Any


class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""


class ModelNotAvailableError(ImageProcessingError):
    """Raised when a required model is not available."""


def _get_yolo_model_info() -> str:
    """Get detailed YOLO model information."""
    try:
        model_name = YOLO_MODEL_PATH.name

        # Get file size if exists
        size_info = ""
        if YOLO_MODEL_PATH.exists():
            size_mb = YOLO_MODEL_PATH.stat().st_size / (1024 * 1024)
            size_info = f" ({size_mb:.1f} MB)"

        # Extract architecture from filename (e.g., yolo12x -> YOLO v12 X-large)
        arch_info = ""
        stem = YOLO_MODEL_PATH.stem.lower()
        if "yolo" in stem:
            # Parse version and size variant
            version_size = stem.replace("yolo", "").replace("v", "")
            if version_size:
                # Extract version number and model size (n/s/m/l/x)
                version_num = "".join(c for c in version_size if c.isdigit())
                size_variant = "".join(c for c in version_size if c.isalpha())

                size_map = {
                    "n": "Nano",
                    "s": "Small",
                    "m": "Medium",
                    "l": "Large",
                    "x": "X-Large",
                }
                size_name = size_map.get(size_variant, size_variant.upper())

                if version_num and size_variant:
                    arch_info = f" | v{version_num} {size_name}"

        return f"{model_name}{size_info}{arch_info} | 80 classes"
    except Exception as e:
        logger.warning(f"Could not get YOLO model info: {e}")
        return YOLO_MODEL_PATH.name


def _build_info_header(cropper: ImageCropper, method: str) -> List[str]:
    """Build the header section of info text."""
    width, height = cropper.original_dimensions
    info_lines = [
        "=" * INFO_SEPARATOR_WIDTH,
        "IMAGE ANALYSIS",
        "=" * INFO_SEPARATOR_WIDTH,
        f"Original dimensions: {width} x {height} pixels",
        f"Aspect ratio: {width / height:.{DEFAULT_ASPECT_RATIO_PRECISION}f}:1",
        "",
    ]

    # Add YOLO model info if using YOLO method (before detection message)
    if method == "yolo":
        info_lines.append(f"Model: {_get_yolo_model_info()}")
        info_lines.append("")

    info_lines.append(f"Detecting object using {method} method...")

    return info_lines


def get_logo_base64() -> Optional[str]:
    """Load crop.png and convert to base64 for inline display."""
    try:
        with CROP_LOGO_PATH.open("rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        logger.warning(f"Logo file '{CROP_LOGO_PATH}' not found")
        return None


def process_image(
    image_file: Optional[str],
    method: str,
    object_name: str,
    confidence: float,
    aspect_mode: str,
    custom_aspect_ratio: str,
    padding: int,
    threshold: int = DEFAULT_THRESHOLD,
    selected_index: Optional[int] = None,
    stored_detections: Optional[List[Dict[str, Union[str, float, List[int]]]]] = None,
) -> ProcessingResult:
    """
    Process the image with the selected parameters.

    Args:
        selected_index: Index of the object to select from stored detections (0-based)
        stored_detections: Previously stored detections to avoid re-running detection

    Returns:
        tuple: (visualization_image, cropped_image, info_text, detections_state, dropdown_update)
    """
    logger.info(f"Processing image: {image_file}, method: {method}")
    logger.debug(
        f"Selected index: {selected_index}, has stored detections: {stored_detections is not None}"
    )

    if image_file is None:
        return ProcessingResult(
            visualization=None,
            cropped_image=None,
            info_text="⚠️ Please upload an image first.",
            detections=None,
            dropdown_update=gr.update(visible=False),
        )

    try:
        # Validate image format
        img_path = Path(image_file)
        if img_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            return ProcessingResult(
                visualization=None,
                cropped_image=None,
                info_text="❌ Error: Image must be JPEG or PNG format.",
                detections=None,
                dropdown_update=gr.update(visible=False),
            )

        # Create cropper instance
        cropper = ImageCropper(image_file, debug=False)
        cropper.load_image()

        # Build info text
        info_lines = _build_info_header(cropper, method)

        # Prepare target objects
        target_objects = [object_name] if object_name and object_name.strip() else None
        all_detections = []
        bounds = None
        detected_label = None
        detected_confidence = None

        # Use stored detections if available and selected_index is provided
        if stored_detections is not None and selected_index is not None:
            all_detections = stored_detections
            info_lines.append("Using previously detected objects...")
        else:
            # Run detection based on method
            if method == "contour":
                bounds = cropper.find_object_bounds_contour(threshold)
                detected_label = "Object"
            elif method == "saliency":
                bounds = cropper.find_object_bounds_saliency()
                detected_label = "Salient Region"
            elif method == "edge":
                bounds = cropper.find_object_bounds_edge()
                detected_label = "Edge-Detected Object"
            elif method == "grabcut":
                bounds = cropper.find_object_bounds_grabcut()
                detected_label = "Foreground Object"
            elif method == "detr":
                if not DETR_AVAILABLE:
                    return ProcessingResult(
                        visualization=None,
                        cropped_image=None,
                        info_text="❌ Error: DETR requires 'transformers' and 'torch'. Install with: pip install transformers torch",
                        detections=None,
                        dropdown_update=gr.update(visible=False),
                    )
                all_detections = cropper.find_all_objects_detr(
                    target_objects, confidence
                )
            elif method == "rt-detr":
                if not RTDETR_AVAILABLE:
                    return ProcessingResult(
                        visualization=None,
                        cropped_image=None,
                        info_text="❌ Error: RT-DETR requires 'transformers' and 'torch'. Install with: pip install transformers torch",
                        detections=None,
                        dropdown_update=gr.update(visible=False),
                    )
                all_detections = cropper.find_all_objects_rtdetr(
                    target_objects, confidence
                )
            elif method == "yolo":
                if not ULTRALYTICS_AVAILABLE:
                    return ProcessingResult(
                        visualization=None,
                        cropped_image=None,
                        info_text="❌ Error: YOLO requires 'ultralytics'. Install with: pip install ultralytics",
                        detections=None,
                        dropdown_update=gr.update(visible=False),
                    )
                all_detections = cropper.find_all_objects_yolo(
                    target_objects, confidence
                )

        # Handle detections
        if all_detections and len(all_detections) > 0:
            # Log detection info
            detection_summary = [
                f"{obj['label']}({obj['confidence']:.2f})" for obj in all_detections
            ]
            logger.debug(f"All detected objects: {detection_summary}")
            if target_objects:
                logger.debug(f"Looking for: {target_objects}")

            # If selected_index is specified, use that detection
            if selected_index is not None and 0 <= selected_index < len(all_detections):
                selected_obj = all_detections[selected_index]
            else:
                # Auto-select best detection
                selected_obj = cropper.select_best_detection(all_detections)

            # Check if we got a valid selection
            if selected_obj is not None:
                bounds = tuple(selected_obj["box"])
                detected_label = selected_obj["label"]
                detected_confidence = selected_obj["confidence"]
            else:
                logger.warning(
                    "select_best_detection returned None despite having detections"
                )
                all_detections = []  # Reset to trigger fallback

        if not all_detections or bounds is None:
            # Fallback to contour if no detections
            info_lines.append("No objects detected, falling back to contour method")
            bounds = cropper.find_object_bounds_contour()
            detected_label = "Object"

        # Safety check: ensure bounds is set
        if bounds is None:
            info_lines.append("⚠️ Warning: No bounds detected, using full image")
            bounds = (
                0,
                0,
                cropper.original_dimensions[0],
                cropper.original_dimensions[1],
            )
            detected_label = "Full Image"

        info_lines.append(f"Initial bounds: {bounds}")

        # Add detection summary
        if all_detections:
            info_lines.append("")
            info_lines.append(f"All Detected Objects ({len(all_detections)}):")
            sorted_detections = sorted(
                all_detections, key=lambda x: x["confidence"], reverse=True
            )
            for i, det in enumerate(sorted_detections, 1):
                info_lines.append(f"  {i}. {det['label']}: {det['confidence']:.2f}")

        # Add selected object info
        info_lines.append("")
        if detected_label:
            if detected_confidence is not None:
                info_lines.append(
                    f"✓ Selected: {detected_label} (confidence: {detected_confidence:.2f})"
                )
            else:
                info_lines.append(f"✓ Selected: {detected_label}")

            if target_objects:
                info_lines.append(f"  (Searched for: {', '.join(target_objects)})")

            if len(all_detections) > 1:
                info_lines.append(
                    "  💡 Multiple objects detected - use dropdown above to select different object"
                )

        # Add padding if requested
        if padding > 0:
            bounds = cropper.add_padding(bounds, padding)
            info_lines.append(f"Bounds with {padding}% padding: {bounds}")

        # Adjust for aspect ratio based on selected mode
        if aspect_mode == "original":
            bounds = cropper.adjust_crop_for_aspect_ratio(bounds, None)
            info_lines.append(f"Bounds with original aspect ratio: {bounds}")
        elif aspect_mode == "custom":
            if custom_aspect_ratio and custom_aspect_ratio.strip():
                try:
                    # Parse aspect ratio (e.g., "16:9", "4:3", "1.5")
                    if ":" in custom_aspect_ratio:
                        width, height = map(float, custom_aspect_ratio.split(":"))
                        target_ratio = width / height
                    else:
                        target_ratio = float(custom_aspect_ratio)

                    bounds = cropper.adjust_crop_for_aspect_ratio(bounds, target_ratio)
                    info_lines.append(
                        f"Bounds with custom aspect ratio {custom_aspect_ratio} ({target_ratio:.2f}): {bounds}"
                    )
                except (ValueError, ZeroDivisionError):
                    info_lines.append(
                        f"⚠️ Invalid aspect ratio format: {custom_aspect_ratio}. Using detected bounds."
                    )
            else:
                info_lines.append(
                    "⚠️ Custom aspect ratio selected but no value provided. Using detected bounds."
                )

        # Debug logging
        logger.debug(f"Final bounds = {bounds}")
        logger.debug(
            f"Final all_detections count = {len(all_detections) if all_detections else 0}"
        )
        if all_detections:
            logger.debug(f"First detection = {all_detections[0]}")
        logger.debug(
            f"Using stored_detections = {stored_detections is not None}"
        )  # Create visualization using the ImageCropper method
        try:
            logger.debug("About to create visualization...")
            logger.debug(f"bounds = {bounds}")
            logger.debug(
                f"all_detections count = {len(all_detections) if all_detections else 0}"
            )

            # Pass None instead of empty list for non-AI detection methods
            detections_for_viz = all_detections if all_detections else None

            # Determine selected_idx: use provided index, or find which detection matches bounds
            if all_detections:
                if selected_index is not None:
                    selected_idx = selected_index
                else:
                    # Find which detection's box matches the selected bounds
                    selected_idx = None
                    for i, det in enumerate(all_detections):
                        if tuple(det["box"]) == bounds:
                            selected_idx = i
                            break
            else:
                selected_idx = None

            logger.debug(f"detections_for_viz = {detections_for_viz is not None}")
            logger.debug(f"selected_idx = {selected_idx}")

            vis_image = cropper.visualize_detections(
                detections_for_viz, selected_idx, bounds
            )

            if vis_image is None:
                raise ValueError("visualize_detections returned None")

            logger.debug(f"vis_image shape = {vis_image.shape}")

            # Convert BGR to RGB for display
            vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
            logger.debug(f"vis_image_rgb shape = {vis_image_rgb.shape}")
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            traceback.print_exc()

            # Return an error instead of raising to see what happens
            return ProcessingResult(
                visualization=None,
                cropped_image=None,
                info_text=f"❌ Visualization error: {str(e)}",
                detections=None,
                dropdown_update=gr.update(visible=False),
            )

        # Create cropped image
        try:
            logger.debug(f"About to crop image with bounds {bounds}")
            pil_image = Image.open(image_file)
            logger.debug(f"PIL image size = {pil_image.size}")
            cropped = pil_image.crop(bounds)
            logger.debug(f"cropped size = {cropped.size}")
        except Exception as e:
            logger.error(f"Cropping error: {e}")
            traceback.print_exc()

            # Return an error instead of raising
            return ProcessingResult(
                visualization=vis_image_rgb if "vis_image_rgb" in locals() else None,
                cropped_image=None,
                info_text=f"❌ Cropping error: {str(e)}",
                detections=all_detections if all_detections else None,
                dropdown_update=gr.update(visible=False),
            )

        # Add final results to info
        crop_width = bounds[2] - bounds[0]
        crop_height = bounds[3] - bounds[1]

        info_lines.extend(
            [
                "",
                "=" * INFO_SEPARATOR_WIDTH,
                "CROP COORDINATES",
                "=" * INFO_SEPARATOR_WIDTH,
                f"Left: {bounds[0]}, Upper: {bounds[1]}, Right: {bounds[2]}, Lower: {bounds[3]}",
                f"Crop dimensions: {crop_width} x {crop_height} pixels",
                f"Crop aspect ratio: {crop_width / crop_height:.{DEFAULT_ASPECT_RATIO_PRECISION}f}:1",
                "=" * INFO_SEPARATOR_WIDTH,
                "✅ Processing complete!",
            ]
        )

        info_text = "\n".join(info_lines)

        # Prepare dropdown update
        dropdown_choices = []
        dropdown_visible = False
        dropdown_value = None

        if len(all_detections) > 1:
            dropdown_visible = True
            for i, det in enumerate(all_detections):
                dropdown_choices.append(
                    f"{i}: {det['label']} ({det['confidence']:.2f})"
                )

            # Set default dropdown value to the selected object
            if selected_index is not None and 0 <= selected_index < len(all_detections):
                dropdown_value = dropdown_choices[selected_index]
            else:
                # Find which object was auto-selected
                for i, det in enumerate(all_detections):
                    if det["box"] == list(bounds):
                        dropdown_value = dropdown_choices[i]
                        break

        logger.debug(
            f"Processing successful: vis_shape={vis_image_rgb.shape}, "
            f"crop_size={cropped.size}, detections={len(all_detections) if all_detections else 0}"
        )

        return ProcessingResult(
            visualization=vis_image_rgb,
            cropped_image=cropped,
            info_text=info_text,
            detections=all_detections if all_detections else None,
            dropdown_update=gr.update(
                choices=dropdown_choices, value=dropdown_value, visible=dropdown_visible
            ),
        )

    except Exception as e:
        error_msg = f"❌ Error processing image: {str(e)}"
        traceback.print_exc()
        return ProcessingResult(
            visualization=None,
            cropped_image=None,
            info_text=error_msg,
            detections=None,
            dropdown_update=gr.update(choices=[], value=None, visible=False),
        )


def batch_crop_objects(
    image_file: Optional[str],
    method: str,
    object_name: str,
    confidence: float,
    aspect_mode: str,
    custom_aspect_ratio: str,
    padding: int,
    threshold: int,
) -> Tuple[List[str], str]:
    """
    Crop all detected objects from an image and return them as downloadable files.
    This is a UI wrapper that handles Gradio-specific concerns.

    Args:
        image_file: Path to input image
        method: Detection method (yolo or detr)
        object_name: Target object name (empty for all objects)
        confidence: Confidence threshold
        aspect_mode: Aspect ratio mode
        custom_aspect_ratio: Custom aspect ratio string
        padding: Padding percentage
        threshold: Binary threshold (for contour method)

    Returns:
        Tuple of (list of file paths for download, status message)
    """
    logger.info(f"Batch cropping objects from: {image_file}, method: {method}")

    if image_file is None:
        return [], "❌ No image provided"

    if method not in ["yolo", "detr", "rt-detr"]:
        return (
            [],
            "❌ Batch crop only works with YOLO, DETR, or RT-DETR detection methods",
        )

    try:
        # Load image and get detections
        cropper = ImageCropper(image_file)
        cropper.load_image()

        # Convert object_name to list format
        target_objects = [object_name] if object_name and object_name.strip() else None

        # Get all detections using appropriate method
        if method == "yolo":
            if not ULTRALYTICS_AVAILABLE:
                return (
                    [],
                    "❌ Error: YOLO requires 'ultralytics'. Install with: pip install ultralytics",
                )
            all_detections = cropper.find_all_objects_yolo(target_objects, confidence)
        elif method == "detr":
            if not DETR_AVAILABLE:
                return (
                    [],
                    "❌ Error: DETR requires 'transformers' and 'torch'. Install with: pip install transformers torch",
                )
            all_detections = cropper.find_all_objects_detr(target_objects, confidence)
        elif method == "rt-detr":
            if not RTDETR_AVAILABLE:
                return (
                    [],
                    "❌ Error: RT-DETR requires 'transformers' and 'torch'. Install with: pip install transformers torch",
                )
            all_detections = cropper.find_all_objects_rtdetr(target_objects, confidence)
        else:
            return [], "❌ Unsupported method for batch crop"

        if not all_detections or len(all_detections) == 0:
            return [], "❌ No objects detected to crop"

        # Parse aspect ratio
        target_aspect_ratio = None
        if aspect_mode == "original":
            target_aspect_ratio = None  # Will use original image aspect
        elif aspect_mode == "custom" and custom_aspect_ratio:
            try:
                if ":" in custom_aspect_ratio:
                    w, h = map(float, custom_aspect_ratio.split(":"))
                    target_aspect_ratio = w / h
                else:
                    target_aspect_ratio = float(custom_aspect_ratio)
            except (ValueError, ZeroDivisionError):
                pass  # Keep None if parsing fails

        # Use the core batch crop method from ImageCropper
        base_name = Path(image_file).stem
        cropped_files = cropper.batch_crop_detections(
            detections=all_detections,
            output_dir=BATCH_OUTPUT_DIR,
            base_filename=base_name,
            padding_percent=padding,
            target_aspect_ratio=target_aspect_ratio,
            image_quality=BATCH_IMAGE_QUALITY,
        )

        status_msg = f"✅ Successfully cropped {len(cropped_files)} object(s). Files ready for download."
        return cropped_files, status_msg

    except Exception as e:
        logger.error(f"Batch crop error: {e}")
        return [], f"❌ Error during batch crop: {str(e)}"


def create_ui() -> gr.Blocks:
    """Create and configure the Gradio interface."""

    # Check available methods
    available_methods = ["contour", "saliency", "edge", "grabcut"]
    if DETR_AVAILABLE:
        available_methods.append("detr")
    if RTDETR_AVAILABLE:
        available_methods.append("rt-detr")
    if ULTRALYTICS_AVAILABLE:
        available_methods.append("yolo")

    # Default method
    default_method = "yolo" if ULTRALYTICS_AVAILABLE else "contour"

    # Load logo
    logo_base64 = get_logo_base64()

    with gr.Blocks(title="AI Image Cropper") as demo:
        if logo_base64:
            gr.Markdown(
                f"""
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 1rem;">
                    <img src="data:image/png;base64,{logo_base64}" alt="Crop Icon" style="width: {LOGO_SIZE}px; height: {LOGO_SIZE}px;">
                    <h1 style="margin: 0;">AI Image Cropper: Image Cropping with AI Detection</h1>
                </div>
                """
            )
        else:
            gr.Markdown("# AI Image Cropper: Image Cropping with AI Detection")

        gr.Markdown(
            "Save time on manual cropping. Upload your images and let AI identify objects, then crop with customizable padding and aspect ratios. Download individual crops or batch process multiple objects at once."
        )

        # Check if sample image exists and show info
        sample_image_path = Path(SAMPLE_IMAGE).absolute()
        # if sample_image_path.exists():
        #     gr.Markdown(
        #         "💡 **Sample image loaded** - Click 'Process Image' to try it out, or upload your own image above."
        #     )

        # State to store all detections
        detections_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input")

                # Image upload with preview (single component)
                default_image_path = Path(SAMPLE_IMAGE).absolute()
                image_input = gr.Image(
                    label="📤 Upload Image (JPEG/PNG/WebP only)",
                    type="filepath",
                    height=UI_IMAGE_HEIGHT,
                    value=(
                        str(default_image_path) if default_image_path.exists() else None
                    ),
                )

                gr.Markdown("### ⚙️ Detection Options")

                # Method selection with info icon
                with gr.Row():
                    method_input = gr.Dropdown(
                        choices=available_methods,
                        value=default_method,
                        label="Detection Method",
                        info="Select the object detection algorithm to use",
                        scale=4,
                    )

                # Confidence threshold
                confidence_input = gr.Slider(
                    minimum=CONFIDENCE_MIN,
                    maximum=CONFIDENCE_MAX,
                    value=DEFAULT_CONFIDENCE,
                    step=CONFIDENCE_STEP,
                    label="Confidence Threshold",
                    info="YOLO/DETR: Minimum confidence for detected objects.",
                )

                # Object selection dropdown (hidden by default, shown when multiple objects detected)
                object_selector = gr.Dropdown(
                    label="Select Detected Object",
                    info="YOLO/DETR: Choose which object to crop when multiple are detected.",
                    visible=False,
                    interactive=True,
                )

                # Object name
                object_input = gr.Textbox(
                    label="Object to Detect (optional)",
                    placeholder="e.g., couch, person, chair",
                    info="YOLO/DETR: Leave empty to detect the largest/most confident object.",
                )

                # Threshold value for contour method
                threshold_input = gr.Slider(
                    minimum=THRESHOLD_MIN,
                    maximum=THRESHOLD_MAX,
                    value=DEFAULT_THRESHOLD,
                    step=THRESHOLD_STEP,
                    label="Binary Threshold",
                    info="Contour: Control foreground/background separation. Lower values may detect more objects.",
                )

                # Aspect ratio mode
                aspect_mode_input = gr.Radio(
                    choices=[
                        ("None (use detected bounds)", "none"),
                        ("Keep Original Aspect Ratio", "original"),
                        ("Custom Aspect Ratio", "custom"),
                    ],
                    value="none",
                    label="Aspect Ratio",
                    info="Choose how to handle the final crop aspect ratio.",
                )

                # Custom aspect ratio input
                custom_aspect_input = gr.Textbox(
                    label="Custom Aspect Ratio",
                    placeholder="e.g., 16:9, 4:3, 1.5, or 2.35",
                    info="Enter ratio as width:height (16:9) or decimal (1.78).",
                    visible=False,
                )

                # Padding
                padding_input = gr.Slider(
                    minimum=PADDING_MIN,
                    maximum=PADDING_MAX,
                    value=DEFAULT_PADDING,
                    step=PADDING_STEP,
                    label="Padding (%)",
                    info="Add padding around the detected object.",
                )

                # Process button
                process_btn = gr.Button(
                    "🚀 Process Image", variant="primary", size="lg"
                )

                # Batch crop button (for YOLO/DETR only)
                batch_crop_btn = gr.Button(
                    "📦 Batch Crop All YOLO/DETR Objects",
                    variant="secondary",
                    size="lg",
                )

                gr.Markdown("### ⬇️ Download Cropped Images")

                # File download component for batch crop
                batch_crop_files = gr.File(
                    label="Cropped Images",
                    file_count="multiple",
                    interactive=False,
                    visible=True,
                )

                # Batch crop status
                batch_status = gr.Textbox(
                    label="Batch Crop Status",
                    interactive=False,
                    visible=False,
                )

            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🔍 Analysis")
                        # Visualization output
                        visualization_output = gr.Image(
                            label="Detection Preview (Green = Selected, Yellow = Other Detections)",
                            type="numpy",
                            show_download_button=True,
                            height=UI_IMAGE_HEIGHT,
                        )

                    with gr.Column():
                        gr.Markdown("### ✅ Result")
                        # Cropped output
                        cropped_output = gr.Image(
                            label="Final Cropped Image",
                            type="pil",
                            show_download_button=True,
                            height=UI_IMAGE_HEIGHT,
                        )

                gr.Markdown("### 📋 Processing Information")

                # Info text
                info_output = gr.Textbox(
                    label="Output",
                    lines=15,
                    max_lines=20,
                    show_copy_button=True,
                )

                gr.Markdown("### 💡 Tips")
                gr.Markdown(
                    """
                    **Getting Started:**
                    - A sample image is loaded by default - just click "Process Image" to try it out!
                    - Upload your own image using the upload area above
                    - The app will automatically process the sample image when it starts

                    **Detection Methods:**
                    - **YOLO** (recommended for AI): Fast and accurate for common objects
                    - **RT-DETR**: Real-time DETR, faster with similar accuracy to DETR
                    - **DETR**: State-of-the-art transformer-based detection
                    - **Contour**: Fast, works well with clear backgrounds
                    - **Saliency**: Identifies visually interesting regions
                    - **Edge**: Fast edge detection with Canny algorithm
                    - **GrabCut**: Precise foreground/background segmentation

                    **Aspect Ratio Options:**
                    - **None**: Use the detected object bounds as-is
                    - **Original**: Maintain the original image's aspect ratio
                    - **Custom**: Specify your own ratio (e.g., 16:9, 4:3, 1.5, 2.35:1)

                    **Saving Images:**
                    - Click the download button (⬇️) on either image to save it
                    - Or right-click on any image and select "Save image as..."

                    For YOLO/RT-DETR/DETR, you can specify objects like: person, car, couch, chair, dog, cat, etc.
                    """
                )

        # Show/hide custom aspect ratio input based on mode selection
        def toggle_aspect_input(mode: str) -> Any:
            return gr.update(visible=(mode == "custom"))

        aspect_mode_input.change(
            fn=toggle_aspect_input,
            inputs=[aspect_mode_input],
            outputs=[custom_aspect_input],
        )

        # Helper function to extract index from dropdown selection
        def on_object_selection_change(
            selection: Optional[str],
            image_file: Optional[str],
            method: str,
            object_name: str,
            confidence: float,
            aspect_mode: str,
            custom_aspect: str,
            padding: int,
            threshold: int,
            stored_detections: Optional[List[Dict[str, Union[str, float, List[int]]]]],
        ) -> ProcessingResult:
            """Called when user selects a different object from dropdown."""
            if selection is None or stored_detections is None:
                return ProcessingResult(
                    visualization=None,
                    cropped_image=None,
                    info_text="⚠️ No selection available.",
                    detections=stored_detections,
                    dropdown_update=gr.update(),
                )

            # Extract index from dropdown format "0: label (0.95)"
            try:
                selected_index = int(selection.split(":")[0])
                return process_image(
                    image_file,
                    method,
                    object_name,
                    confidence,
                    aspect_mode,
                    custom_aspect,
                    padding,
                    threshold,
                    selected_index=selected_index,
                    stored_detections=stored_detections,
                )
            except (ValueError, IndexError) as e:
                return ProcessingResult(
                    visualization=None,
                    cropped_image=None,
                    info_text=f"❌ Error parsing selection: {e}",
                    detections=stored_detections,
                    dropdown_update=gr.update(),
                )

        # Create wrapper function for button click (ensures proper parameter passing)
        def process_image_from_button(
            img: Optional[str],
            method: str,
            obj: str,
            conf: float,
            aspect: str,
            custom: str,
            pad: int,
            thresh: int,
        ) -> ProcessingResult:
            """Wrapper function for button click to ensure correct parameter passing."""
            return process_image(
                img,
                method,
                obj,
                conf,
                aspect,
                custom,
                pad,
                thresh,
                selected_index=None,
                stored_detections=None,
            )

        # Connect the button to the processing function
        process_btn.click(
            fn=process_image_from_button,
            inputs=[
                image_input,
                method_input,
                object_input,
                confidence_input,
                aspect_mode_input,
                custom_aspect_input,
                padding_input,
                threshold_input,
            ],
            outputs=[
                visualization_output,
                cropped_output,
                info_output,
                detections_state,
                object_selector,
            ],
        )

        # Connect batch crop button
        def batch_crop_with_status(
            img: Optional[str],
            method: str,
            obj: str,
            conf: float,
            aspect: str,
            custom: str,
            pad: int,
            thresh: int,
        ) -> Tuple[List[str], Any]:
            """Wrapper to run batch crop and return files for download."""
            files, status = batch_crop_objects(
                img, method, obj, conf, aspect, custom, pad, thresh
            )
            return files, gr.update(value=status, visible=True)

        batch_crop_btn.click(
            fn=batch_crop_with_status,
            inputs=[
                image_input,
                method_input,
                object_input,
                confidence_input,
                aspect_mode_input,
                custom_aspect_input,
                padding_input,
                threshold_input,
            ],
            outputs=[batch_crop_files, batch_status],
        )

        # Handle object selection from dropdown
        object_selector.change(
            fn=on_object_selection_change,
            inputs=[
                object_selector,
                image_input,
                method_input,
                object_input,
                confidence_input,
                aspect_mode_input,
                custom_aspect_input,
                padding_input,
                threshold_input,
                detections_state,
            ],
            outputs=[
                visualization_output,
                cropped_output,
                info_output,
                detections_state,
                object_selector,
            ],
        )

        # Auto-process sample image on startup if it exists
        sample_image_path = Path(SAMPLE_IMAGE).absolute()
        if sample_image_path.exists():

            def auto_process_sample() -> ProcessingResult:
                """Auto-process the sample image when the interface loads."""
                return process_image_from_button(
                    str(sample_image_path),
                    default_method,
                    "",  # no specific object
                    DEFAULT_CONFIDENCE,
                    "none",  # aspect mode
                    "",  # custom aspect
                    DEFAULT_PADDING,
                    DEFAULT_THRESHOLD,
                )

            # Trigger auto-processing on interface load
            demo.load(
                fn=auto_process_sample,
                outputs=[
                    visualization_output,
                    cropped_output,
                    info_output,
                    detections_state,
                    object_selector,
                ],
            )

    return demo


def main() -> None:
    """Launch the Gradio interface."""
    print("=" * 60)
    print("Image Cropper UI - Starting...")
    print("=" * 60)
    print(f"YOLO Available: {ULTRALYTICS_AVAILABLE}")
    print(f"DETR Available: {DETR_AVAILABLE}")
    print("=" * 60)

    if not ULTRALYTICS_AVAILABLE:
        print("⚠️  Warning: YOLO not available. Install with: uv add ultralytics")

    if not DETR_AVAILABLE:
        print("⚠️  Warning: DETR not available. Install with: uv add transformers torch")

    print("\nLaunching Gradio interface...")

    demo.launch(
        share=SHARE_PUBLICLY,
        server_name=SERVER_HOST,
        server_port=SERVER_PORT,
    )


# Create demo at module level for Gradio's auto-reload
demo = create_ui()

if __name__ == "__main__":
    main()
