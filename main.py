import logging
from typing import List, Tuple, Union
import math
import cv2
import numpy as np
import onnxruntime as ort
import requests
import tempfile
from huggingface_hub import hf_hub_download

# Constants
REPO_ID = "Longcat2957/dwpose-onnx"

eps = 0.01

def draw_bodypose(canvas, candidate, subset):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18],
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas


def draw_handpose(canvas, all_hand_peaks):
    import matplotlib

    H, W, C = canvas.shape

    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    # (person_number*2, 21, 2)
    for i in range(len(all_hand_peaks)):
        peaks = all_hand_peaks[i]
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255,
                    thickness=2,
                )

        for _, keyponit in enumerate(peaks):
            x, y = keyponit

            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas, all_lmks):
    H, W, C = canvas.shape
    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas


def draw_pose(pose, height: int, width: int, include_face: bool = True, include_hands: bool = True) -> np.ndarray:
    canvas = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    candidate = pose["bodies"]
    subset = pose["body_scores"]
    canvas = draw_bodypose(canvas, candidate, subset)

    if include_face:
        faces = pose["faces"]
        canvas = draw_facepose(canvas, faces)

    if include_hands:
        hands = pose["hands"]
        canvas = draw_handpose(canvas, hands)

    return canvas


def nms(boxes: np.ndarray, scores: np.ndarray, nms_thr: float) -> List[int]:
    """Perform Non-Maximum Suppression (NMS) for a single class.

    Args:
        boxes (np.ndarray): Bounding boxes of shape (N, 4).
        scores (np.ndarray): Scores for each bounding box of shape (N,).
        nms_thr (float): IoU threshold for NMS.

    Returns:
        List[int]: Indices of selected bounding boxes after NMS.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(
    boxes: np.ndarray, scores: np.ndarray, nms_thr: float, score_thr: float
) -> Union[np.ndarray, None]:
    """Perform multi-class NMS (class-aware).

    Args:
        boxes (np.ndarray): Bounding boxes of shape (N, 4).
        scores (np.ndarray): Scores of shape (N, num_classes).
        nms_thr (float): IoU threshold for NMS.
        score_thr (float): Score threshold for filtering.

    Returns:
        np.ndarray | None:
            - If not empty, shape is (M, 6) with columns [x1, y1, x2, y2, score, class].
            - If no boxes remain, returns None.
    """
    final_dets = []
    num_classes = scores.shape[1]

    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue

        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]

        keep = nms(valid_boxes, valid_scores, nms_thr)
        if len(keep) > 0:
            cls_inds = np.full((len(keep), 1), cls_ind, dtype=np.float32)
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], cls_inds], axis=1
            )
            final_dets.append(dets)

    if len(final_dets) == 0:
        return None

    return np.concatenate(final_dets, axis=0)


def bbox_xyxy2cs(
    bbox: np.ndarray, padding: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert bounding box (x1, y1, x2, y2) to center & scale format.

    Args:
        bbox (np.ndarray): Bounding box of shape (4,) or (N, 4) in [x1, y1, x2, y2].
        padding (float): Scale padding factor. Defaults to 1.0.

    Returns:
        (center, scale):
        - center (np.ndarray): Shape (2,) or (N, 2), the center of the bbox (x, y).
        - scale (np.ndarray): Shape (2,) or (N, 2), the scale (w, h) of the bbox.
    """
    # If single bbox, reshape to (1, 4)
    if bbox.ndim == 1:
        bbox = bbox[None, :]

    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if len(center) == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """Adjust the bbox scale to match the given aspect ratio (width/height).

    Args:
        bbox_scale (np.ndarray): Bbox scale in shape (2,) -> (w, h).
        aspect_ratio (float): Desired aspect ratio, width / height.

    Returns:
        np.ndarray: Adjusted bbox scale in shape (2,).
    """
    w, h = np.hsplit(bbox_scale, [1])
    if (w > h * aspect_ratio).any():
        return np.hstack([w, w / aspect_ratio])
    else:
        return np.hstack([h * aspect_ratio, h])


def _rotate_point(point: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a 2D point by a given angle in radians.

    Args:
        point (np.ndarray): Shape (2,).
        angle_rad (float): Angle in radians.

    Returns:
        np.ndarray: Rotated point in shape (2,).
    """
    sin_val, cos_val = np.sin(angle_rad), np.cos(angle_rad)
    rotation_matrix = np.array([[cos_val, -sin_val], [sin_val, cos_val]])
    return rotation_matrix @ point


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the third point to form an affine transform reference.

    Given two points (a, b), the third point is formed by rotating the vector
    (a - b) by 90 degrees about b.

    Args:
        a (np.ndarray): Shape (2,).
        b (np.ndarray): Shape (2,).

    Returns:
        np.ndarray: Third point in shape (2,).
    """
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]


def get_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot_deg: float,
    output_size: Tuple[int, int],
    shift: Tuple[float, float] = (0.0, 0.0),
    inv: bool = False,
) -> np.ndarray:
    """Compute the affine transform matrix for warping a region to output_size.

    Args:
        center (np.ndarray): Shape (2,) - bounding box center (x, y).
        scale (np.ndarray): Shape (2,) - bounding box scale (w, h).
        rot_deg (float): Rotation angle in degrees.
        output_size (Tuple[int, int]): Output image size (width, height).
        shift (Tuple[float, float]): Shift factor (relative to bbox width/height). Defaults to (0.0, 0.0).
        inv (bool): If True, compute the inverse transform (dst->src). Default is False (src->dst).

    Returns:
        np.ndarray: 2x3 affine transform matrix.
    """
    shift = np.array(shift)
    src_w = scale[0]
    dst_w, dst_h = output_size

    rot_rad = np.deg2rad(rot_deg)
    src_dir = _rotate_point(np.array([0.0, src_w * -0.5]), rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = [dst_w * 0.5, dst_h * 0.5] + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        warp_mat = cv2.getAffineTransform(dst, src)
    else:
        warp_mat = cv2.getAffineTransform(src, dst)

    return warp_mat


def top_down_affine(
    input_size: Tuple[int, int],
    bbox_scale: np.ndarray,
    bbox_center: np.ndarray,
    image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop the image to the bounding box area via affine transform.

    Args:
        input_size (Tuple[int, int]): Target (width, height) for the model input.
        bbox_scale (np.ndarray): BBox scale in shape (2,).
        bbox_center (np.ndarray): BBox center in shape (2,).
        image (np.ndarray): Original image (H, W, C).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - The cropped (affine-warped) image of shape (input_size[1], input_size[0], 3).
            - The updated bbox scale in shape (2,).
    """
    target_w, target_h = input_size

    # Adjust aspect ratio
    bbox_scale = _fix_aspect_ratio(bbox_scale, aspect_ratio=target_w / target_h)

    # Calculate affine transform matrix
    warp_mat = get_warp_matrix(
        center=bbox_center,
        scale=bbox_scale,
        rot_deg=0,
        output_size=(target_w, target_h),
    )

    # Warp the image
    warp_size = (int(target_w), int(target_h))
    warped_image = cv2.warpAffine(image, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

    return warped_image, bbox_scale


def get_simcc_maximum(
    simcc_x: np.ndarray, simcc_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Obtain maximum heatmap positions (x, y) from SimCC arrays.

    Args:
        simcc_x (np.ndarray): (N, K, Wx) - x-axis SimCC.
        simcc_y (np.ndarray): (N, K, Wy) - y-axis SimCC.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - locations (np.ndarray): (N, K, 2) max value locations (x, y).
            - vals (np.ndarray): (N, K) max values.
    """
    # simcc_x, simcc_y shape = (N, K, Wx or Wy)
    N, K, Wx = simcc_x.shape

    # Reshape to (N*K, Wx)
    simcc_x_flat = simcc_x.reshape(N * K, -1)
    simcc_y_flat = simcc_y.reshape(N * K, -1)

    # Argmax
    x_locs = np.argmax(simcc_x_flat, axis=1)
    y_locs = np.argmax(simcc_y_flat, axis=1)

    # Gather locations
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.max(simcc_x_flat, axis=1)
    max_val_y = np.max(simcc_y_flat, axis=1)

    # Combine values
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x

    # If val <= 0, set location = -1
    locs[vals <= 0.0] = -1

    # Reshape back
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals


def decode(
    simcc_x: np.ndarray, simcc_y: np.ndarray, simcc_split_ratio: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode x-axis and y-axis SimCC into keypoints and scores.

    Args:
        simcc_x (np.ndarray): (N, K, Wx) - x-axis SimCC.
        simcc_y (np.ndarray): (N, K, Wy) - y-axis SimCC.
        simcc_split_ratio (float): Ratio for scaling down keypoint predictions.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - keypoints (np.ndarray): (N, K, 2) keypoint coordinates.
            - scores (np.ndarray): (N, K) confidence scores.
    """
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints /= simcc_split_ratio

    return keypoints, scores


class DWposeDetector:
    """DWposeDetector is a wrapper class that performs:
       1) Object detection using YOLOX
       2) Pose estimation using DWpose
    """

    def __init__(self, device: str = "cuda", verbose: bool = False) -> None:
        """Initialize the DWposeDetector.

        Args:
            device (str): Target device ('cpu', 'cuda', or 'cuda:device_id'). Defaults to 'cuda'.
            verbose (bool): If True, set logging level to INFO. Otherwise WARNING. Defaults to False.
        """
        self.logger = logging.getLogger("DWposeDetector")
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

        # Add handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info("Initializing DWposeDetector")

        # Download ONNX models from Hugging Face Hub
        self.det_model_path = hf_hub_download(repo_id=REPO_ID, filename="yolox_l.onnx")
        self.pose_model_path = hf_hub_download(repo_id=REPO_ID, filename="dw-ll_ucoco_384.onnx")

        # Configure Inference Sessions
        providers, provider_options = self.check_device(device)
        self.det_session = ort.InferenceSession(
            self.det_model_path,
            providers=providers,
            provider_options=provider_options,
        )
        self.pose_session = ort.InferenceSession(
            self.pose_model_path,
            providers=providers,
            provider_options=provider_options,
        )

    def check_device(self, device: str) -> Tuple[List[str], List[dict]]:
        """Check if the device is available and return the appropriate ONNXRuntime providers.

        Args:
            device (str): Device string (e.g., 'cpu', 'cuda', 'cuda:1', etc.).

        Returns:
            Tuple[List[str], List[dict]]:
                - providers (List[str]): Provider names for onnxruntime.
                - provider_options (List[dict]): Additional provider-specific options.
        """
        available_providers = ort.get_available_providers()

        if device.lower() == "cpu":
            self.logger.info("Using CPU as requested")
            return ["CPUExecutionProvider"], [{}]

        # CUDA device check
        if "CUDAExecutionProvider" in available_providers:
            if ":" in device:
                try:
                    gpu_id = int(device.split(":")[1])
                    self.logger.info(f"CUDA is available and will use GPU ID: {gpu_id}")
                    return ["CUDAExecutionProvider", "CPUExecutionProvider"], [
                        {"device_id": gpu_id},
                        {},
                    ]
                except (ValueError, IndexError):
                    self.logger.warning(
                        f"Invalid GPU ID format in '{device}'. Using default GPU (0)."
                    )
                    return ["CUDAExecutionProvider", "CPUExecutionProvider"], [
                        {"device_id": 0},
                        {},
                    ]
            else:
                self.logger.info("CUDA is available and will use default GPU (0)")
                return ["CUDAExecutionProvider", "CPUExecutionProvider"], [
                    {"device_id": 0},
                    {},
                ]
        else:
            self.logger.warning("CUDA is not available. Falling back to CPU.")
            return ["CPUExecutionProvider"], [{}]

    @staticmethod
    def guess_input(input_source: str) -> np.ndarray:
        """Load an image from a file path or URL into a 3-channel BGR numpy array.

        Args:
            input_source (str): Local file path or URL (e.g., 'http://...', 'https://...').

        Returns:
            np.ndarray: Image in 3-channel BGR format.

        Raises:
            TypeError: If input_source is not a string.
            ValueError: If loading or reading the image fails.
        """
        if not isinstance(input_source, str):
            raise TypeError(f"Input must be a string. Got: {type(input_source)}")

        # Attempt to handle URL
        if input_source.startswith(("http://", "https://")):
            # Extract extension from URL
            url_path = input_source.split("?")[0]  # remove query parameters
            extension = url_path.split(".")[-1] if "." in url_path else "tmp"
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=f".{extension}", delete=True
                ) as temp_file:
                    response = requests.get(input_source, stream=True)
                    response.raise_for_status()

                    # Save to a temporary file
                    with open(temp_file.name, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    image_path = temp_file.name
            except Exception as e:
                raise ValueError(f"Failed to load image from URL: {input_source}\n{e}")
        else:
            image_path = input_source

        # Read the image
        try:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Could not read image file: {image_path}")

            # Convert to 3-channel BGR if needed
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:  # BGRA -> BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            elif len(image.shape) == 3 and image.shape[2] != 3:
                # Convert to RGB then back to BGR as a fallback
                image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2BGR)

            return image
        except Exception as e:
            raise ValueError(f"Error processing image: {image_path}\n{e}")

    @staticmethod
    def preprocess_det(
        image: np.ndarray,
        input_shape: Tuple[int, int] = (640, 640),
        swap: Tuple[int, int, int] = (2, 0, 1),
    ) -> Tuple[np.ndarray, float]:
        """Preprocess an image for the YOLOX detector.

        1) Letterbox resize to input_shape (default: 640x640).
        2) Normalization or type casting if needed.
        3) Change channel order based on `swap` (default: (2,0,1)).

        Args:
            image (np.ndarray): Original image (H, W, C).
            input_shape (Tuple[int, int]): Model input size (height, width).
            swap (Tuple[int, int, int]): Channel swap order.

        Returns:
            (preprocessed_image, resize_ratio):
            - preprocessed_image (np.ndarray): Shape (C, input_shape[0], input_shape[1]).
            - resize_ratio (float): Scale factor used for resizing.
        """
        padded_image = np.full(
            (input_shape[0], input_shape[1], 3), 114, dtype=np.uint8
        )

        r = min(input_shape[0] / image.shape[0], input_shape[1] / image.shape[1])
        resized_image = cv2.resize(
            image,
            (int(image.shape[1] * r), int(image.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        new_h, new_w = resized_image.shape[:2]
        padded_image[:new_h, :new_w] = resized_image

        # Channel swap
        padded_image = padded_image.transpose(swap)
        padded_image = padded_image.astype(np.float32)
        return padded_image, r

    @staticmethod
    def preprocess_pose(
        image: np.ndarray,
        boxes: np.ndarray,
        input_shape: Tuple[int, int] = (192, 256),
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Preprocess the image for the pose model (e.g., RTMPose).

        This includes:
        1) Transforming each detected bounding box to a fixed-size rectangle via affine transform.
        2) Normalizing pixel values with mean/std.

        Args:
            image (np.ndarray): Original image (H, W, C).
            boxes (np.ndarray): Detected bounding boxes in xyxy format (N, 4).
            input_shape (Tuple[int, int]): Desired input size (width, height) for the pose model.

        Returns:
            Tuple of:
            - out_images (List[np.ndarray]): List of cropped images (each of shape input_shape).
            - centers (List[np.ndarray]): List of centers for each bbox (each shape (2,)).
            - scales (List[np.ndarray]): List of scales for each bbox (each shape (2,)).
        """
        # If no boxes, use the entire image as a single bbox
        if len(boxes) == 0:
            boxes = np.array([[0, 0, image.shape[1], image.shape[0]]])

        preprocessed_images = []
        centers = []
        scales = []

        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        for box in boxes:
            # xyxy format
            x0, y0, x1, y1 = box
            bbox = np.array([x0, y0, x1, y1])

            center, scale = bbox_xyxy2cs(bbox, padding=1.25)
            warped_img, scale = top_down_affine(input_shape, scale, center, image)

            # Normalize
            warped_img = (warped_img - mean) / std
            preprocessed_images.append(warped_img)
            centers.append(center)
            scales.append(scale)

        return preprocessed_images, centers, scales

    def inference_detector(self, image: np.ndarray) -> np.ndarray:
        """Run the YOLOX detector on the image.

        Args:
            image (np.ndarray): Original image array (H, W, C).

        Returns:
            np.ndarray: Filtered bounding boxes in xyxy format.
        """
        self.logger.info("Running inference on the detector model")

        # Preprocess
        input_shape = (640, 640)
        input_data, ratio = self.preprocess_det(image, input_shape)

        # Inference
        det_input_name = self.det_session.get_inputs()[0].name
        ort_inputs = {det_input_name: input_data[None, :, :, :]}
        ort_output = self.det_session.run(None, ort_inputs)

        # Postprocess
        predictions = self.postprocess_detector(ort_output[0], input_shape, p6=False)[0]

        # Convert box format from cx, cy, w, h -> x1, y1, x2, y2
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)

        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio

        # NMS
        detections = multiclass_nms(
            boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1
        )
        if detections is not None:
            final_boxes, final_scores, final_cls_inds = (
                detections[:, :4],
                detections[:, 4],
                detections[:, 5],
            )
            # Filter by threshold and class 0 (person class in COCO)
            score_mask = final_scores > 0.3
            class_mask = final_cls_inds == 0
            combined_mask = np.logical_and(score_mask, class_mask)
            final_boxes = final_boxes[combined_mask]
        else:
            final_boxes = np.array([])

        return final_boxes

    @staticmethod
    def postprocess_detector(
        ort_output: Union[np.ndarray, List[np.ndarray]],
        input_shape: Tuple[int, int],
        p6: bool = False,
    ) -> np.ndarray:
        """Post-process YOLOX outputs.

        Args:
            ort_output (np.ndarray | List[np.ndarray]): Model raw output.
            input_shape (Tuple[int, int]): Input shape used for inference.
            p6 (bool): If True, use [8, 16, 32, 64] strides. Otherwise [8, 16, 32].

        Returns:
            np.ndarray: Processed detections of shape (N, M, 85) where M could be different by stride.
        """
        if isinstance(ort_output, list):
            ort_output = ort_output[0]

        # Strides
        strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]
        hsizes = [input_shape[0] // s for s in strides]
        wsizes = [input_shape[1] // s for s in strides]

        grids = []
        expanded_strides = []

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, axis=1)
        expanded_strides = np.concatenate(expanded_strides, axis=1)

        ort_output[..., :2] = (ort_output[..., :2] + grids) * expanded_strides
        ort_output[..., 2:4] = np.exp(ort_output[..., 2:4]) * expanded_strides
        return ort_output

    @staticmethod
    def postprocess_pose(
        ort_outputs: List[List[np.ndarray]],
        model_input_size: Tuple[int, int],
        centers: List[np.ndarray],
        scales: List[np.ndarray],
        simcc_split_ratio: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Post-process pose model outputs (SimCC).

        Args:
            ort_outputs (List[List[np.ndarray]]): Raw outputs for each person. Each element is a list of [simcc_x, simcc_y].
            model_input_size (Tuple[int, int]): (width, height) used for model input.
            centers (List[np.ndarray]): List of bounding box centers for each person.
            scales (List[np.ndarray]): List of bounding box scales for each person.
            simcc_split_ratio (float): Ratio to scale keypoint predictions. Default is 2.0.

        Returns:
            (keypoints, scores):
            - keypoints (np.ndarray): (N, K, 2) final keypoint coordinates.
            - scores (np.ndarray): (N, K) final keypoint confidence scores.
        """
        all_keypoints = []
        all_scores = []

        w, h = model_input_size

        for i, outputs in enumerate(ort_outputs):
            simcc_x, simcc_y = outputs
            keypoints, scores = decode(simcc_x, simcc_y, simcc_split_ratio)

            # Rescale keypoints to original image space
            keypoints = (keypoints / [w, h]) * scales[i] + (centers[i] - scales[i] / 2.0)

            all_keypoints.append(keypoints[0])
            all_scores.append(scores[0])

        return np.array(all_keypoints), np.array(all_scores)

    def inference_pose(
        self, image: np.ndarray, boxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run pose estimation on detected bounding boxes.

        Args:
            image (np.ndarray): Original image (H, W, C).
            boxes (np.ndarray): Bounding boxes in xyxy format (N, 4).

        Returns:
            (keypoints, scores):
            - keypoints (np.ndarray): (N, K, 2).
            - scores (np.ndarray): (N, K).
        """
        self.logger.info("Running inference on the pose model")

        # The model input shape from the ONNX graph
        pose_input_name = self.pose_session.get_inputs()[0].name
        _, _, pose_h, pose_w = self.pose_session.get_inputs()[0].shape
        model_input_size = (pose_w, pose_h)

        # Preprocess
        cropped_images, centers, scales = self.preprocess_pose(image, boxes, model_input_size)

        # Inference for each bounding box
        results = []
        out_names = [o.name for o in self.pose_session.get_outputs()]

        for cropped_img in cropped_images:
            # Shape -> (C, H, W)
            pose_input = cropped_img.transpose(2, 0, 1)
            ort_inputs = {pose_input_name: [pose_input]}
            ort_output = self.pose_session.run(out_names, ort_inputs)
            # ort_output is a list: [simcc_x, simcc_y]
            results.append(ort_output)

        # Postprocess
        keypoints, scores = self.postprocess_pose(results, model_input_size, centers, scales)
        return keypoints, scores

    def _format_pose(self, candidates, scores, width, height):
        num_candidates, _, locs = candidates.shape

        candidates[..., 0] /= float(width)
        candidates[..., 1] /= float(height)

        bodies = candidates[:, :18].copy()
        bodies = bodies.reshape(num_candidates * 18, locs)

        body_scores = scores[:, :18]
        for i in range(len(body_scores)):
            for j in range(len(body_scores[i])):
                if body_scores[i][j] > 0.3:
                    body_scores[i][j] = int(18 * i + j)
                else:
                    body_scores[i][j] = -1

        faces = candidates[:, 24:92]
        faces_scores = scores[:, 24:92]

        hands = np.vstack([candidates[:, 92:113], candidates[:, 113:]])
        hands_scores = np.vstack([scores[:, 92:113], scores[:, 113:]])

        pose = dict(
            bodies=bodies,
            body_scores=body_scores,
            hands=hands,
            hands_scores=hands_scores,
            faces=faces,
            faces_scores=faces_scores,
        )

        return pose

    def __call__(self, input_source: str, output_path: str | None) -> Tuple[np.ndarray, np.ndarray]:
        """Perform detection + pose estimation on a given input image source.

        Args:
            input_source (str): Image file path or URL.

        Returns:
            (keypoints, scores):
            - keypoints (np.ndarray): Shape (N, K, 2).
            - scores (np.ndarray): Shape (N, K).
        """
        self.logger.info(f"Start Inference on {input_source}")
        image = self.guess_input(input_source)

        # 1) Detection
        final_boxes = self.inference_detector(image)
        if final_boxes.size == 0:
            self.logger.warning("No boxes detected. Will use entire image as a single bbox.")

        # 2) Pose Estimation
        keypoints, scores = self.inference_pose(image, final_boxes)

        # Merge keypoints & scores (shape: (N, K, 3)) then insert 'neck'
        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)

        # Compute an approximate neck keypoint
        # neck is average of L-shoulder and R-shoulder if both are confident
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # Confidence for neck if both shoulders are above 0.3
        neck[:, 2] = np.logical_and(
            keypoints_info[:, 5, 2] > 0.3,
            keypoints_info[:, 6, 2] > 0.3
        ).astype(float)

        # Insert neck at index 17
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)

        # Remap to OpenPose index order if needed
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]

        # Final keypoints & scores
        final_keypoints = new_keypoints_info[..., :2]
        final_scores = new_keypoints_info[..., 2]

        # Format pose data
        pose_data = self._format_pose(
            final_keypoints, final_scores, image.shape[1], image.shape[0]
        )
        self.logger.info("Inference completed successfully")
        
        # Save the pose data if output path is provided
        if output_path:
            pose_canvas = draw_pose(pose_data, image.shape[0], image.shape[1])
            cv2.imwrite(output_path, pose_canvas)
            self.logger.info(f"Pose visualization saved to {output_path}")
        # Return keypoints and scores

        return pose_data

# 사용 예시
if __name__ == "__main__":
    # DWposeDetector 객체 생성
    detector = DWposeDetector(device="cuda:0", verbose=True)  # GPU 사용 시 "cuda"로 변경
    
    # 이미지 경로
    image_path = "sample.png"
    
    # 포즈 추출
    pose_data = detector(image_path, output_path="output_555.png")
    
