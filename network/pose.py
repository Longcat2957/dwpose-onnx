import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any

from network.base import ONNXWrapper


class PoseEstimator(ONNXWrapper):
    """
    PoseEstimator class is a wrapper for ONNX-based pose estimation models.
    """

    def __init__(
        self,
        device: str = "cuda",
        verbose: bool = False,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.3,
        simcc_split_ratio: float = 2.0,
    ) -> None:
        """
        Initialize the PoseEstimator.

        Args:
            device (str): Target device ('cpu', 'cuda', or 'cuda:device_id'). Defaults to 'cuda'.
            verbose (bool): If True, set logging level to INFO. Otherwise WARNING. Defaults to False.
            repo_id (str, optional): Hugging Face Hub repository ID. Required if using Hugging Face.
            filename (str, optional): Filename of the model in the repository. Required if using Hugging Face.
            model_path (str, optional): Direct path to the model file. Used if repo_id and filename are not provided.
            conf_threshold (float): Confidence threshold for keypoints. Defaults to 0.3.
            model_input_shape (Tuple[int, int]): Model input size (width, height). Defaults to (192, 256).
            simcc_split_ratio (float): Ratio for scaling down keypoint predictions. Defaults to 2.0.
        """
        super().__init__(device, verbose, repo_id, filename, model_path)

        self.conf_threshold = conf_threshold
        self.simcc_split_ratio = simcc_split_ratio

    def infer(
        self, image: np.ndarray, boxes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run pose estimation on detected bounding boxes.

        Args:
            image (np.ndarray): Original image (H, W, C).
            boxes (np.ndarray): Bounding boxes in xyxy format (N, 4).

        Returns:
            tuple:
                - keypoints (np.ndarray): (N, K, 2) keypoint coordinates normalized to [0-1].
                - scores (np.ndarray): (N, K) keypoint confidence scores.
        """
        self.logger.info(f"Running pose estimation on {len(boxes)} bounding boxes")

        # Preprocess
        cropped_images, centers, scales = self.preprocess(image, boxes)

        # Inference for each bounding box
        results = []

        for cropped_img in cropped_images:
            # Shape -> (C, H, W)
            pose_input = cropped_img.transpose(2, 0, 1)

            # Run inference
            ort_inputs = {self.input_names[0]: [pose_input]}
            ort_output = self.session.run(self.output_names, ort_inputs)

            # ort_output is a list: [simcc_x, simcc_y]
            results.append(ort_output)

        # Postprocess
        keypoints, scores = self.postprocess(results, centers, scales)

        return keypoints, scores

    def preprocess(
        self, image: np.ndarray, boxes: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        Preprocess the image for pose estimation.

        Args:
            image (np.ndarray): Original image (H, W, C) in BGR format.
            boxes (np.ndarray): Detected bounding boxes in xyxy format (N, 4).

        Returns:
            tuple:
                - preprocessed_images (list[np.ndarray]): List of cropped and normalized images.
                - centers (list[np.ndarray]): List of bbox centers (each shape (2,)).
                - scales (list[np.ndarray]): List of bbox scales (each shape (2,)).
        """
        self.logger.info("Preprocessing images for pose estimation")

        # If no boxes, use the entire image as a single bbox
        if len(boxes) == 0:
            boxes = np.array([[0, 0, image.shape[1], image.shape[0]]])

        preprocessed_images = []
        centers = []
        scales = []

        # Normalization values
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        for box in boxes:
            # xyxy format
            x0, y0, x1, y1 = box
            bbox = np.array([x0, y0, x1, y1])

            # Get center and scale
            center, scale = self._bbox_xyxy2cs(bbox, padding=1.25)

            # Apply affine transformation
            warped_img, scale = self._top_down_affine(
                self.model_input_shape, scale, center, image
            )

            # Normalize image
            warped_img = (warped_img - mean) / std

            # Store results
            preprocessed_images.append(warped_img)
            centers.append(center)
            scales.append(scale)

        return preprocessed_images, centers, scales

    def _bbox_xyxy2cs(
        self, bbox: np.ndarray, padding: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert bounding box (x1, y1, x2, y2) to center & scale format.

        Args:
            bbox (np.ndarray): Bounding box of shape (4,) in [x1, y1, x2, y2].
            padding (float): Scale padding factor. Defaults to 1.0.

        Returns:
            tuple:
                - center (np.ndarray): Shape (2,), the center of the bbox (x, y).
                - scale (np.ndarray): Shape (2,), the scale (w, h) of the bbox.
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

    def _fix_aspect_ratio(
        self, bbox_scale: np.ndarray, aspect_ratio: float
    ) -> np.ndarray:
        """
        Adjust the bbox scale to match the given aspect ratio (width/height).

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

    def _rotate_point(self, point: np.ndarray, angle_rad: float) -> np.ndarray:
        """
        Rotate a 2D point by a given angle in radians.

        Args:
            point (np.ndarray): Shape (2,).
            angle_rad (float): Angle in radians.

        Returns:
            np.ndarray: Rotated point in shape (2,).
        """
        sin_val, cos_val = np.sin(angle_rad), np.cos(angle_rad)
        rotation_matrix = np.array([[cos_val, -sin_val], [sin_val, cos_val]])
        return rotation_matrix @ point

    def _get_3rd_point(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute the third point to form an affine transform reference.

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

    def _get_warp_matrix(
        self,
        center: np.ndarray,
        scale: np.ndarray,
        rot_deg: float,
        output_size: tuple[int, int],
        shift: tuple[float, float] = (0.0, 0.0),
        inv: bool = False,
    ) -> np.ndarray:
        """
        Compute the affine transform matrix for warping a region to output_size.

        Args:
            center (np.ndarray): Shape (2,) - bounding box center (x, y).
            scale (np.ndarray): Shape (2,) - bounding box scale (w, h).
            rot_deg (float): Rotation angle in degrees.
            output_size (tuple[int, int]): Output image size (width, height).
            shift (tuple[float, float]): Shift factor (relative to bbox width/height). Defaults to (0.0, 0.0).
            inv (bool): If True, compute the inverse transform (dst->src). Default is False (src->dst).

        Returns:
            np.ndarray: 2x3 affine transform matrix.
        """
        shift = np.array(shift)
        src_w = scale[0]
        dst_w, dst_h = output_size

        rot_rad = np.deg2rad(rot_deg)
        src_dir = self._rotate_point(np.array([0.0, src_w * -0.5]), rot_rad)
        dst_dir = np.array([0.0, dst_w * -0.5])

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale * shift
        src[1, :] = center + src_dir + scale * shift
        src[2, :] = self._get_3rd_point(src[0, :], src[1, :])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = [dst_w * 0.5, dst_h * 0.5] + dst_dir
        dst[2, :] = self._get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            warp_mat = cv2.getAffineTransform(dst, src)
        else:
            warp_mat = cv2.getAffineTransform(src, dst)

        return warp_mat

    def _top_down_affine(
        self,
        input_size: tuple[int, int],
        bbox_scale: np.ndarray,
        bbox_center: np.ndarray,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Crop the image to the bounding box area via affine transform.

        Args:
            input_size (tuple[int, int]): Target (width, height) for the model input.
            bbox_scale (np.ndarray): BBox scale in shape (2,).
            bbox_center (np.ndarray): BBox center in shape (2,).
            image (np.ndarray): Original image (H, W, C).

        Returns:
            tuple:
                - warped_image (np.ndarray): The cropped image of shape (input_size[1], input_size[0], 3).
                - bbox_scale (np.ndarray): The updated bbox scale in shape (2,).
        """
        target_w, target_h = input_size

        # Adjust aspect ratio
        bbox_scale = self._fix_aspect_ratio(
            bbox_scale, aspect_ratio=target_w / target_h
        )

        # Calculate affine transform matrix
        warp_mat = self._get_warp_matrix(
            center=bbox_center,
            scale=bbox_scale,
            rot_deg=0,
            output_size=(target_w, target_h),
        )

        # Warp the image
        warp_size = (int(target_w), int(target_h))
        warped_image = cv2.warpAffine(
            image, warp_mat, warp_size, flags=cv2.INTER_LINEAR
        )

        return warped_image, bbox_scale

    def postprocess(
        self,
        ort_outputs: list[list[np.ndarray]],
        centers: list[np.ndarray],
        scales: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Post-process pose model outputs (SimCC).

        Args:
            ort_outputs (list[list[np.ndarray]]): Raw outputs for each person. Each element is a list of [simcc_x, simcc_y].
            centers (list[np.ndarray]): List of bounding box centers for each person.
            scales (list[np.ndarray]): List of bounding box scales for each person.

        Returns:
            tuple:
                - keypoints (np.ndarray): (N, K, 2) final keypoint coordinates.
                - scores (np.ndarray): (N, K) final keypoint confidence scores.
        """
        self.logger.info("Post-processing pose estimation outputs")

        all_keypoints = []
        all_scores = []

        w, h = self.model_input_shape

        for i, outputs in enumerate(ort_outputs):
            simcc_x, simcc_y = outputs
            keypoints, scores = self._decode(simcc_x, simcc_y, self.simcc_split_ratio)

            # Rescale keypoints to original image space
            keypoints = (keypoints / [w, h]) * scales[i] + (
                centers[i] - scales[i] / 2.0
            )

            all_keypoints.append(keypoints[0])
            all_scores.append(scores[0])

        return np.array(all_keypoints), np.array(all_scores)

    def _get_simcc_maximum(
        self, simcc_x: np.ndarray, simcc_y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Obtain maximum heatmap positions (x, y) from SimCC arrays.

        Args:
            simcc_x (np.ndarray): (N, K, Wx) - x-axis SimCC.
            simcc_y (np.ndarray): (N, K, Wy) - y-axis SimCC.

        Returns:
            tuple:
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

    def _decode(
        self, simcc_x: np.ndarray, simcc_y: np.ndarray, simcc_split_ratio: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Decode x-axis and y-axis SimCC into keypoints and scores.

        Args:
            simcc_x (np.ndarray): (N, K, Wx) - x-axis SimCC.
            simcc_y (np.ndarray): (N, K, Wy) - y-axis SimCC.
            simcc_split_ratio (float): Ratio for scaling down keypoint predictions.

        Returns:
            tuple:
                - keypoints (np.ndarray): (N, K, 2) keypoint coordinates.
                - scores (np.ndarray): (N, K) confidence scores.
        """
        keypoints, scores = self._get_simcc_maximum(simcc_x, simcc_y)
        keypoints /= simcc_split_ratio

        return keypoints, scores
