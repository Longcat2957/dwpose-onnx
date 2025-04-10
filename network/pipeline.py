import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np

from .det import ObjectDetector
from .pose import PoseEstimator


class DWPosePipeline:
    """
    DWPosePipeline 클래스는 ObjectDetector와 PoseEstimator를 통합하여
    전체 포즈 추정 파이프라인을 제공합니다.
    """

    def __init__(
        self,
        device: str = "cuda",
        verbose: bool = False,
        det_repo_id: Optional[str] = None,
        det_filename: Optional[str] = None,
        det_conf_threshold: float = 0.3,
        det_nms_threshold: float = 0.45,
        pose_repo_id: Optional[str] = None,
        pose_filename: Optional[str] = None,
        pose_conf_threshold: float = 0.3,
        simcc_split_ratio: float = 2.0,
    ) -> None:
        """
        DWPosePipeline 초기화

        Args:
            device (str): 타겟 디바이스 ('cpu', 'cuda', or 'cuda:device_id'). 기본값은 'cuda'.
            verbose (bool): True이면 로깅 레벨을 INFO로 설정, 아니면 WARNING. 기본값은 False.
            det_repo_id (str, optional): 디텍터 모델의 Hugging Face Hub 레포지토리 ID.
            det_filename (str, optional): 디텍터 모델의 파일명.
            det_conf_threshold (float): 객체 감지 신뢰도 임계값. 기본값은 0.3.
            det_nms_threshold (float): 객체 감지 NMS 임계값. 기본값은 0.45.
            pose_repo_id (str, optional): 포즈 모델의 Hugging Face Hub 레포지토리 ID.
            pose_filename (str, optional): 포즈 모델의 파일명.
            pose_conf_threshold (float): 키포인트 신뢰도 임계값. 기본값은 0.3.
            simcc_split_ratio (float): 키포인트 예측 스케일링 비율. 기본값은 2.0.
        """
        # 로거 설정
        self.logger = logging.getLogger("DWPosePipeline")
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

        # 핸들러가 없으면 추가
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info("DWPosePipeline 초기화 중")

        # 객체 감지기 초기화
        self.detector = ObjectDetector(
            device=device,
            verbose=verbose,
            repo_id=det_repo_id,
            filename=det_filename,
            conf_threshold=det_conf_threshold,
            nms_threshold=det_nms_threshold,
        )

        # 포즈 추정기 초기화
        self.pose_estimator = PoseEstimator(
            device=device,
            verbose=verbose,
            repo_id=pose_repo_id,
            filename=pose_filename,
            conf_threshold=pose_conf_threshold,
            simcc_split_ratio=simcc_split_ratio,
        )

        self.logger.info("DWPosePipeline 초기화 완료")

    def __call__(self, image: np.ndarray) -> Dict[str, Any]:
        """
        이미지에서 객체 감지 및 포즈 추정 수행

        Args:
            image (np.ndarray): 입력 이미지 (H, W, C) 형식의 BGR 이미지.

        Returns:

        """
        self.logger.info("객체 감지 및 포즈 추정 시작")

        # 객체 감지
        boxes = self.detector(image)

        # 포즈 추정
        keypoints, scores = self.pose_estimator(image, boxes)

        self.logger.info("객체 감지 및 포즈 추정 완료")

        # 이제 여기 후처리 부분이 필요함
        keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1
        )  # (number_of_people, 133, 3)
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)  # (1, 3)
        neck[:, 2] = np.logical_and(
            keypoints_info[:, 5, 2] > self.pose_estimator.conf_threshold,
            keypoints_info[:, 6, 2] > self.pose_estimator.conf_threshold,
        ).astype(np.float32)

        # Insert neck keypoint
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1
        )  # (number_of_people, 134, 3)

        # 순서를 바꿔준다. mmpose -> OpenPose
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]

        # Final keypoints & scores
        number_of_people = new_keypoints_info.shape[0]
        final_keypoints = new_keypoints_info[..., :2]  # (number_of_people, 134, 2)
        final_scores = new_keypoints_info[..., 2]  # (number_of_people, 134)

        # normalize keypoints
        final_keypoints[..., 0] /= float(image.shape[1])
        final_keypoints[..., 1] /= float(image.shape[0])

        # body, hand, face로 분리
        body_keypoints = final_keypoints[:, :18]  # (number_of_people, 18, 2)
        left_foot_keypoints = final_keypoints[:, 18:21]  # (number_of_people, 3, 2)
        right_foot_keypoints = final_keypoints[:, 21:24]  # (number_of_people, 3, 2)
        face_keypoints = final_keypoints[:, 24:92]  # (number_of_people, 68, 2)
        left_hand_keypoints = final_keypoints[:, 92:113]  # (number_of_people, 21, 2)
        right_hand_keypoints = final_keypoints[:, 113:]  # (number_of_people, 21, 2)

        # 키포인트와 신뢰도 점수 합치기 (x, y, score)
        body_keypoints_with_scores = np.concatenate(
            [body_keypoints, final_scores[:, :18, None]], axis=2
        )  # (number_of_people, 18, 3)

        left_foot_keypoints_with_scores = np.concatenate(
            [left_foot_keypoints, final_scores[:, 18:21, None]], axis=2
        )  # (number_of_people, 3, 3)

        right_foot_keypoints_with_scores = np.concatenate(
            [right_foot_keypoints, final_scores[:, 21:24, None]], axis=2
        )  # (number_of_people, 3, 3)

        face_keypoints_with_scores = np.concatenate(
            [face_keypoints, final_scores[:, 24:92, None]], axis=2
        )  # (number_of_people, 68, 3)

        left_hand_keypoints_with_scores = np.concatenate(
            [left_hand_keypoints, final_scores[:, 92:113, None]], axis=2
        )  # (number_of_people, 21, 3)

        right_hand_keypoints_with_scores = np.concatenate(
            [right_hand_keypoints, final_scores[:, 113:, None]], axis=2
        )  # (number_of_people, 21, 3)

        return {
            "number_of_people": number_of_people,  # int
            "boxes": boxes,  # (number_of_people, 4)
            "keypoints": {
                "body": body_keypoints_with_scores,  # (number_of_people, 18, 3)
                "left_foot": left_foot_keypoints_with_scores,  # (number_of_people, 3, 3)
                "right_foot": right_foot_keypoints_with_scores,  # (number_of_people, 3, 3)
                "face": face_keypoints_with_scores,  # (number_of_people, 68, 3)
                "left_hand": left_hand_keypoints_with_scores,  # (number_of_people, 21, 3)
                "right_hand": right_hand_keypoints_with_scores,  # (number_of_people, 21, 3)
            },
        }
