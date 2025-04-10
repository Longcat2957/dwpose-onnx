import cv2
from PIL import Image
import math
import numpy as np
import logging
from typing import Any, Optional
import matplotlib.colors as mplc


# temp function (to be removed)
def write_image(obj: np.ndarray, filename: str) -> None:
    """
    이미지를 파일로 저장하는 함수입니다.

    Args:
        obj (np.ndarray): 저장할 이미지 (numpy 배열)
        filename (str): 저장할 파일 이름
    """
    cv2.imwrite(filename, obj)
    return None


class PoseVisualizer:
    def __init__(
        self,
        stick_width: int = 4,
        point_radius: int = 4,
        face_point_radius: int = 3,
        conf_threshold: float = 0.3,
        alpha: float = 0.6,
        draw_body: bool = True,
        draw_hands: bool = True,
        draw_face: bool = True,
        draw_feet: bool = True,
        verbose: bool = True,
    ):
        """
        PoseVisualizer 초기화

        Args:
            stick_width (int): 관절 연결선의 두께. 기본값은 4.
            point_radius (int): 키포인트 원의 반지름. 기본값은 4.
            conf_threshold (float): 키포인트 표시 신뢰도 임계값. 기본값은 0.3.
            alpha (float): 포즈 오버레이 불투명도 (0-1). 기본값은 0.6.
            draw_body (bool): 몸체 키포인트 표시 여부. 기본값은 True.
            draw_hands (bool): 손 키포인트 표시 여부. 기본값은 True.
            draw_face (bool): 얼굴 키포인트 표시 여부. 기본값은 True.
            draw_feet (bool): 발 키포인트 표시 여부. 기본값은 True.
        """
        # 로거 설정
        self.logger = logging.getLogger("PoseVisualizer")
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        # 핸들러가 없으면 추가
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # 시각화 스타일 매개변수
        self.stick_width = stick_width
        self.point_radius = point_radius
        self.face_point_radius = face_point_radius
        self.conf_threshold = conf_threshold
        self.alpha = alpha

        # 시각화 옵션
        self.draw_body = draw_body
        self.draw_hands = draw_hands
        self.draw_face = draw_face
        self.draw_feet = draw_feet

        # 안전 마진 - 화면 밖으로 나가는 점들을 필터링하기 위한 값
        self.eps = 0.01

        # 신체 관절 연결 정보 및 색상
        self.BODY_LIMB_SEQ = [
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
            # [3, 17],
            # [6, 18],
        ]

        # 관절 연결선 색상
        self.LIMB_COLORS = [
            [255, 0, 0],  # 빨강
            [255, 85, 0],  # 주황
            [255, 170, 0],  # 주황-노랑
            [255, 255, 0],  # 노랑
            [170, 255, 0],  # 연두
            [85, 255, 0],  # 연두-초록
            [0, 255, 0],  # 초록
            [0, 255, 85],  # 초록-청록
            [0, 255, 170],  # 청록
            [0, 255, 255],  # 하늘
            [0, 170, 255],  # 하늘-파랑
            [0, 85, 255],  # 파랑
            [0, 0, 255],  # 진파랑
            [85, 0, 255],  # 파랑-보라
            [170, 0, 255],  # 보라
            [255, 0, 255],  # 자주
            # [255, 0, 170],  # 자주-핑크
            # [255, 0, 85],  # 핑크
        ]

        # 손 관절 연결 정보
        self.HAND_SEQ = [
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

        # 발 관절 연결 정보
        self.FOOT_SEQ = [
            [0, 2],
            [1, 2],
        ]

    def __call__(
        self,
        input_image: np.ndarray,
        pose_data: dict[str, Any],
        draw_body: Optional[bool] = None,
        draw_hands: Optional[bool] = None,
        draw_face: Optional[bool] = None,
        draw_feet: Optional[bool] = None,
        draw_background: Optional[bool] = None,
        format: str = "np",
    ) -> np.ndarray:
        """
        포즈 데이터를 이미지에 시각화하여 반환합니다.

        Args:
            input_image (np.ndarray): 원본 입력 이미지


        return canvas

            pose_data (dict[str, Any]): 포즈 데이터 딕셔너리
            draw_body (bool): 몸체 키포인트 표시 여부. 기본값은 True.
            draw_hands (bool): 손 키포인트 표시 여부. 기본값은 True.
            draw_face (bool): 얼굴 키포인트 표시 여부. 기본값은 True.
            draw_feet (bool): 발 키포인트 표시 여부. 기본값은 True.
            draw_background (bool): 배경 이미지를 그릴지 여부. 기본값은 False.
            format (str): 반환 이미지 형식 (np 또는 pil). 기본값은 "np".

        Returns:
            np.ndarray: 포즈가 시각화된 이미지


        ### 참고 pose_data의 형식

        키포인트의 각 값들은 0-1 정규화 되어 있음

        ```json
        {
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
        ```

        """
        assert format in ["np", "pil"], "format은 'np' 또는 'pil'이어야 합니다."

        draw_body = draw_body if draw_body is not None else self.draw_body
        draw_hands = draw_hands if draw_hands is not None else self.draw_hands
        draw_face = draw_face if draw_face is not None else self.draw_face
        draw_feet = draw_feet if draw_feet is not None else self.draw_feet
        draw_background = draw_background if draw_background is not None else False

        if draw_background:
            canvas = input_image.copy()
        else:
            canvas = np.zeros_like(input_image)

        if draw_body:
            self.logger.info(f"trying to draw body")
            body_keypoints = pose_data["keypoints"]["body"]
            canvas = self._draw_body(canvas, body_keypoints)
            # DEBUG !! must be removed
            # write_image(canvas, "body.jpg")

        if draw_hands:
            self.logger.info(f"trying to draw hands")
            left_hand_keypoints = pose_data["keypoints"]["left_hand"]
            right_hand_keypoints = pose_data["keypoints"]["right_hand"]
            canvas = self._draw_hands(canvas, left_hand_keypoints, right_hand_keypoints)
            # DEBUG !! must be removed
            # write_image(canvas, "hands.jpg")

        if draw_face:
            self.logger.info(f"trying to draw face")
            face_keypoints = pose_data["keypoints"]["face"]
            canvas = self._draw_face(canvas, face_keypoints)
            # DEBUG !! must be removed
            # write_image(canvas, "face.jpg")

        if draw_feet:
            self.logger.info(f"trying to draw feet")
            left_foot_keypoints = pose_data["keypoints"]["left_foot"]
            right_foot_keypoints = pose_data["keypoints"]["right_foot"]
            canvas = self._draw_foot(canvas, left_foot_keypoints, right_foot_keypoints)
            # DEBUG !! must be removed
            # write_image(canvas, "feet.jpg")

        if format == "np":
            return canvas
        elif format == "pil":
            # PIL로 변환
            pil_image = Image.fromarray(canvas)
            return pil_image

    def _is_valid_point(self, point: np.ndarray, H: int, W: int) -> bool:
        """
        포인트가 유효한지 확인 (이미지 내부에 있고 마진을 초과)

        Args:
            point (np.ndarray): [x, y] 좌표 (정규화됨 0-1)
            H (int): 이미지 높이
            W (int): 이미지 너비

        Returns:
            bool: 유효한 포인트인지 여부
        """
        x, y = point
        x_px = int(x * W)
        y_px = int(y * H)

        # 이미지 범위 안에 있는지 확인 & 최소 마진(self.eps) 초과 확인
        return 0 <= x_px < W and 0 <= y_px < H and x > self.eps and y > self.eps

    def _draw_body(self, canvas: np.ndarray, body_keypoints: np.ndarray) -> np.ndarray:
        """
        몸체 키포인트와 관절 연결선을 그립니다.

        Args:
            canvas (np.ndarray): 그리기용 캔버스 (H x W x 3)
            body_keypoints (np.ndarray): (number_of_people, 18, 3) 배열.
                각 keypoint는 [x, y, confidence] 값으로, x,y는 0~1 정규화 값입니다.

        Returns:
            np.ndarray: 관절 연결선과 키포인트가 그려진 캔버스.
        """
        H, W, _ = canvas.shape
        num_people = body_keypoints.shape[0]

        # 1. limb(관절 연결선) 그리기
        for person_idx in range(num_people):
            keypoints = body_keypoints[person_idx]  # shape (18, 3)
            for limb_idx, limb in enumerate(self.BODY_LIMB_SEQ):
                # reference 코드와 동일하게 인덱스는 1부터 시작하므로 0-index로 변환합니다.
                kp1 = limb[0] - 1
                kp2 = limb[1] - 1

                # 두 keypoint 모두 confidence가 기준치 이상이어야 함.
                if (
                    keypoints[kp1, 2] < self.conf_threshold
                    or keypoints[kp2, 2] < self.conf_threshold
                ):
                    continue

                # reference 코드에서는 좌표 채널을 뒤바꿔서 사용합니다.
                # 즉, candidate[...,0] * W 를 Y좌표, candidate[...,1] * H 를 X좌표로 사용합니다.
                y1 = keypoints[kp1, 0] * float(W)
                x1 = keypoints[kp1, 1] * float(H)
                y2 = keypoints[kp2, 0] * float(W)
                x2 = keypoints[kp2, 1] * float(H)

                mX = (x1 + x2) / 2
                mY = (y1 + y2) / 2

                # 두 점 사이의 거리를 계산합니다.
                length = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if length < 1:
                    continue  # 너무 짧은 선은 그리지 않습니다.

                # reference 코드에서는 아래와 같이 각도를 계산합니다.
                angle = math.degrees(math.atan2(x1 - x2, y1 - y2))

                # cv2.ellipse2Poly에 전달할 center는 (col, row)이므로 (int(mY), int(mX))를 사용합니다.
                polygon = cv2.ellipse2Poly(
                    (int(mY), int(mX)),  # center (col, row) 순서
                    (int(length / 2), self.stick_width),
                    int(angle),
                    0,
                    360,
                    1,
                )
                color = self.LIMB_COLORS[limb_idx % len(self.LIMB_COLORS)]
                cv2.fillConvexPoly(canvas, polygon, color)

        # 2. reference 코드에서는 limb들을 그린 후에 캔버스를 약간 어둡게 만듭니다.
        #    (여기서는 self.alpha를 곱해줍니다.)
        canvas = (canvas * self.alpha).astype(np.uint8)

        # 3. 각 keypoint(관절 위치)에 원을 그립니다.
        for person_idx in range(num_people):
            keypoints = body_keypoints[person_idx]
            for i in range(18):
                if keypoints[i, 2] < self.conf_threshold:
                    continue
                # circle 그릴 때는 원래 정규화된 좌표 순서대로 사용합니다.
                x = int(keypoints[i, 0] * W)
                y = int(keypoints[i, 1] * H)
                color = self.LIMB_COLORS[i % len(self.LIMB_COLORS)]
                cv2.circle(canvas, (x, y), self.point_radius, color, thickness=-1)

        return canvas

    def _draw_hands(
        self,
        canvas: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
    ) -> np.ndarray:
        """
        왼손과 오른손의 키포인트와 연결선을 그립니다.
        연결선은 cv2.line을 사용하여 단순 선으로 그립니다.

        Args:
            canvas (np.ndarray): 그릴 캔버스 (H x W x 3).
            left_hand (np.ndarray): (number_of_people, 21, 3) 왼손 키포인트 배열.
            right_hand (np.ndarray): (number_of_people, 21, 3) 오른손 키포인트 배열.

        Returns:
            np.ndarray: 손 키포인트와 연결선이 그려진 캔버스.
        """
        H, W, _ = canvas.shape
        num_people = left_hand.shape[0]

        # 왼손 그리기
        for person_idx in range(num_people):
            keypoints = left_hand[person_idx]  # shape (21, 3)

            # 손 연결선 그리기 (cv2.line 사용)
            for i, connection in enumerate(self.HAND_SEQ):
                kp1, kp2 = connection  # HAND_SEQ는 0-indexed로 가정
                # 신뢰도가 기준치 이상인 경우에만 연결
                if (
                    keypoints[kp1, 2] < self.conf_threshold
                    or keypoints[kp2, 2] < self.conf_threshold
                ):
                    continue
                pt1 = (int(keypoints[kp1, 0] * W), int(keypoints[kp1, 1] * H))
                pt2 = (int(keypoints[kp2, 0] * W), int(keypoints[kp2, 1] * H))
                # HSV 컬러값 계산: 전체 HAND_SEQ 길이를 기준으로 hue를 결정
                rgb_color = (
                    mplc.hsv_to_rgb([i / float(len(self.HAND_SEQ)), 1.0, 1.0]) * 255
                )
                color = tuple(rgb_color.astype(int).tolist())
                cv2.line(canvas, pt1, pt2, color, thickness=self.stick_width)

            # 손 키포인트 원 그리기
            for i in range(21):
                if keypoints[i, 2] < self.conf_threshold:
                    continue
                pt = (int(keypoints[i, 0] * W), int(keypoints[i, 1] * H))
                cv2.circle(canvas, pt, self.point_radius, (0, 0, 255), thickness=-1)

        # 오른손 그리기
        for person_idx in range(num_people):
            keypoints = right_hand[person_idx]  # shape (21, 3)

            # 손 연결선 그리기 (cv2.line 사용)
            for i, connection in enumerate(self.HAND_SEQ):
                kp1, kp2 = connection
                if (
                    keypoints[kp1, 2] < self.conf_threshold
                    or keypoints[kp2, 2] < self.conf_threshold
                ):
                    continue
                pt1 = (int(keypoints[kp1, 0] * W), int(keypoints[kp1, 1] * H))
                pt2 = (int(keypoints[kp2, 0] * W), int(keypoints[kp2, 1] * H))
                # 오른손도 동일하게 HSV 색상 계산
                rgb_color = (
                    mplc.hsv_to_rgb([i / float(len(self.HAND_SEQ)), 1.0, 1.0]) * 255
                )
                color = tuple(rgb_color.astype(int).tolist())
                cv2.line(canvas, pt1, pt2, color, thickness=self.stick_width)

            # 손 키포인트 원 그리기
            for i in range(21):
                if keypoints[i, 2] < self.conf_threshold:
                    continue
                pt = (int(keypoints[i, 0] * W), int(keypoints[i, 1] * H))
                cv2.circle(canvas, pt, self.point_radius, (0, 0, 255), thickness=-1)

        return canvas

    def _draw_face(
        self,
        canvas: np.ndarray,
        face_keypoints: np.ndarray,
    ) -> np.ndarray:
        """
        얼굴 키포인트와 연결선을 그립니다.

        Args:
            canvas (np.ndarray): 그릴 캔버스 (H x W x 3).
            face_keypoints (np.ndarray): (number_of_people, 68, 3) 얼굴 키포인트 배열.

        Returns:
            np.ndarray: 얼굴 키포인트와 연결선이 그려진 캔버스.
        """
        H, W, _ = canvas.shape
        num_people = face_keypoints.shape[0]

        # 얼굴 연결선 그리기
        for person_idx in range(num_people):
            keypoints = face_keypoints[person_idx]
            # 얼굴은 점만 찍음 (하얀색)
            for i in range(68):
                if keypoints[i, 2] < self.conf_threshold:
                    continue
                pt = (int(keypoints[i, 0] * W), int(keypoints[i, 1] * H))
                cv2.circle(
                    canvas, pt, self.face_point_radius, (255, 255, 255), thickness=-1
                )

        return canvas

    def _draw_foot(
        self,
        canvas: np.ndarray,
        left_foot: np.ndarray,
        right_foot: np.ndarray,
    ) -> np.ndarray:
        """
        왼발과 오른발의 키포인트와 연결선을 일반적인 OpenPose 시각화 방식으로 그립니다.

        Args:
            canvas (np.ndarray): 그릴 캔버스 (H x W x 3).
            left_foot (np.ndarray): (number_of_people, 3, 3) 왼발 키포인트 배열.
            right_foot (np.ndarray): (number_of_people, 3, 3) 오른발 키포인트 배열.

        Returns:
            np.ndarray: 발 키포인트와 연결선이 그려진 캔버스.
        """
        H, W, _ = canvas.shape
        num_people = left_foot.shape[0]

        # 왼발 그리기
        for person_idx in range(num_people):
            keypoints = left_foot[person_idx]  # shape (3, 3)
            # 발 연결선 그리기 : self.FOOT_SEQ = [[0, 1], [1, 2]]
            for connection in self.FOOT_SEQ:
                kp1, kp2 = connection
                if (
                    keypoints[kp1, 2] < self.conf_threshold
                    or keypoints[kp2, 2] < self.conf_threshold
                ):
                    continue
                pt1 = (int(keypoints[kp1, 0] * W), int(keypoints[kp1, 1] * H))
                pt2 = (int(keypoints[kp2, 0] * W), int(keypoints[kp2, 1] * H))
                # 왼발은 색상 리스트의 첫 번째 색상을 사용 (또는 원하는 색상으로 조정 가능)
                color = self.LIMB_COLORS[0]
                cv2.line(canvas, pt1, pt2, color, thickness=self.stick_width)

            # 왼발 키포인트 원 그리기
            for i in range(3):
                if keypoints[i, 2] < self.conf_threshold:
                    continue
                pt = (int(keypoints[i, 0] * W), int(keypoints[i, 1] * H))
                cv2.circle(canvas, pt, self.point_radius, (0, 0, 255), thickness=-1)

        # 오른발 그리기
        for person_idx in range(num_people):
            keypoints = right_foot[person_idx]  # shape (3, 3)
            # 발 연결선 그리기
            for connection in self.FOOT_SEQ:
                kp1, kp2 = connection
                if (
                    keypoints[kp1, 2] < self.conf_threshold
                    or keypoints[kp2, 2] < self.conf_threshold
                ):
                    continue
                pt1 = (int(keypoints[kp1, 0] * W), int(keypoints[kp1, 1] * H))
                pt2 = (int(keypoints[kp2, 0] * W), int(keypoints[kp2, 1] * H))
                # 오른발은 색상 리스트의 두 번째 색상을 사용 (있지 않으면 첫 번째 색상 사용)
                color = (
                    self.LIMB_COLORS[1]
                    if len(self.LIMB_COLORS) > 1
                    else self.LIMB_COLORS[0]
                )
                cv2.line(canvas, pt1, pt2, color, thickness=self.stick_width)

            # 오른발 키포인트 원 그리기
            for i in range(3):
                if keypoints[i, 2] < self.conf_threshold:
                    continue
                pt = (int(keypoints[i, 0] * W), int(keypoints[i, 1] * H))
                cv2.circle(canvas, pt, self.point_radius, (0, 0, 255), thickness=-1)

        return canvas
