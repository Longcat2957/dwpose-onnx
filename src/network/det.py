import numpy as np
import cv2
from typing import Optional

from src.network.base import ONNXWrapper


class ObjectDetector(ONNXWrapper):
    """
    ObjectDetector class is a wrapper for ONNX-based object detection models.
    """

    def __init__(
        self,
        device: str = "cuda",
        verbose: bool = False,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.3,
        nms_threshold: float = 0.45,
    ) -> None:
        """
        Initialize the ObjectDetector.

        Args:
            device (str): Target device ('cpu', 'cuda', or 'cuda:device_id'). Defaults to 'cuda'.
            verbose (bool): If True, set logging level to INFO. Otherwise WARNING. Defaults to False.
            repo_id (str, optional): Hugging Face Hub repository ID. Required if using Hugging Face.
            filename (str, optional): Filename of the model in the repository. Required if using Hugging Face.
            model_path (str, optional): Direct path to the model file. Used if repo_id and filename are not provided.
            conf_threshold (float): Confidence threshold for detections. Defaults to 0.3.
            nms_threshold (float): IoU threshold for non-maximum suppression. Defaults to 0.45.
        """
        super().__init__(device, verbose, repo_id, filename, model_path)

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Run inference on the input image.

        Args:
            image (np.ndarray): Input image in BGR format (H, W, C).

        Returns:
            np.ndarray: Detected bounding boxes in xyxy format (N, 4).
        """
        self.logger.info("Running object detection inference")

        # Preprocess image
        input_data, ratio = self.preprocess(image)

        # Prepare input for ONNX session
        input_name = self.input_names[0]
        ort_inputs = {input_name: input_data[None, :, :, :]}

        # Run inference
        ort_output = self.session.run(None, ort_inputs)

        # Postprocess results
        boxes = self.postprocess(ort_output[0], ratio)

        return boxes

    def preprocess(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Preprocess the input image for the model.

        Args:
            image (np.ndarray): Input image as a NumPy array (H, W, C).

        Returns:
            tuple:
                - preprocessed_image (np.ndarray): Shape (C, input_shape[0], input_shape[1]).
                - resize_ratio (float): Scale factor used for resizing.
        """
        # Create padded image filled with value 114
        padded_image = np.full(
            (self.model_input_shape[0], self.model_input_shape[1], 3),
            114,
            dtype=np.uint8,
        )

        # Calculate resize ratio
        r = min(
            self.model_input_shape[0] / image.shape[0],
            self.model_input_shape[1] / image.shape[1],
        )

        # Resize the image
        resized_image = cv2.resize(
            image,
            (int(image.shape[1] * r), int(image.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        # Place the resized image in the padded image
        new_h, new_w = resized_image.shape[:2]
        padded_image[:new_h, :new_w] = resized_image

        # Channel swap (HWC -> CHW for PyTorch/ONNX models)
        padded_image = padded_image.transpose((2, 0, 1))

        # Convert to float32
        padded_image = padded_image.astype(np.float32)

        return padded_image, r

    def postprocess(
        self, outputs: np.ndarray, ratio: float, p6: bool = False
    ) -> np.ndarray:
        """
        Post-process YOLOX detection outputs.

        Args:
            outputs (np.ndarray): Raw model outputs.
            ratio (float): Resize ratio used in preprocessing.
            p6 (bool): If True, use [8, 16, 32, 64] strides. Otherwise [8, 16, 32].

        Returns:
            np.ndarray: Processed detections in format [x1, y1, x2, y2, score, class_id].
                    Returns None if no valid detections.
        """
        self.logger.info("Post-processing detection outputs")

        # Strides
        strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]
        hsizes = [self.model_input_shape[0] // s for s in strides]
        wsizes = [self.model_input_shape[1] // s for s in strides]

        # Process outputs
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

        # Adjust predictions
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        # Get predictions from the first batch
        predictions = outputs[0]

        # Convert box format from cx, cy, w, h -> x1, y1, x2, y2
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)

        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio

        # Apply NMS
        detections = self._multiclass_nms(
            boxes_xyxy, scores, self.nms_threshold, self.conf_threshold
        )

        # Filter for person class (0) for pose estimation
        if detections is not None:
            final_boxes, final_scores, final_cls_inds = (
                detections[:, :4],
                detections[:, 4],
                detections[:, 5],
            )
            # Filter by threshold and class 0 (person class in COCO)
            score_mask = final_scores > self.conf_threshold
            class_mask = final_cls_inds == 0  # Person class
            combined_mask = np.logical_and(score_mask, class_mask)
            final_boxes = final_boxes[combined_mask]

            self.logger.info(f"Detected {len(final_boxes)} persons")
            return final_boxes
        else:
            self.logger.info("No detections after NMS")
            return np.array([])

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, nms_thr: float) -> list:
        """
        Perform Non-Maximum Suppression (NMS) for a single class.

        Args:
            boxes (np.ndarray): Bounding boxes of shape (N, 4).
            scores (np.ndarray): Scores for each bounding box of shape (N,).
            nms_thr (float): IoU threshold for NMS.

        Returns:
            list: Indices of selected bounding boxes after NMS.
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

            # Calculate IoU
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

    def _multiclass_nms(
        self, boxes: np.ndarray, scores: np.ndarray, nms_thr: float, score_thr: float
    ) -> np.ndarray:
        """
        Perform multi-class NMS (class-aware).

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

            keep = self._nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.full((len(keep), 1), cls_ind, dtype=np.float32)
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], axis=1
                )
                final_dets.append(dets)

        if len(final_dets) == 0:
            return None

        return np.concatenate(final_dets, axis=0)
