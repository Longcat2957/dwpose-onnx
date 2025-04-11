import os
import logging
import tempfile
import cv2
import numpy as np
import requests


class ImageLoader:
    """
    ImageLoader는 다양한 소스(파일 경로, URL)에서 이미지를 로드하는 유틸리티 클래스입니다.
    로드된 이미지는 항상 3채널 RGB 형식의 NumPy 배열로 반환됩니다.
    """

    def __init__(self, verbose: bool = False):
        """
        ImageLoader 클래스 초기화

        Args:
            verbose (bool): 로깅 레벨을 INFO로 설정할지 여부 (False이면 WARNING)
        """
        self.logger = self._setup_logger(verbose)

    def _setup_logger(self, verbose: bool) -> logging.Logger:
        """
        로거 설정

        Args:
            verbose (bool): 로깅 레벨을 INFO로 설정할지 여부 (False이면 WARNING)

        Returns:
            logging.Logger: 설정된 로거 인스턴스
        """
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO if verbose else logging.WARNING)

        # 핸들러가 없는 경우에만 추가
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def load(self, input_source: str) -> np.ndarray:
        """
        파일 경로 또는 URL에서 이미지 로드

        Args:
            input_source (str): 로컬 파일 경로 또는 URL (e.g., 'http://...', 'https://...')

        Returns:
            np.ndarray: 3채널 RGB 형식의 이미지 배열

        Raises:
            TypeError: input_source가 문자열이 아닌 경우
            ValueError: 이미지 로드 또는 읽기 실패 시
        """
        if not isinstance(input_source, str):
            raise TypeError(
                f"입력은 문자열이어야 합니다. 받은 타입: {type(input_source)}"
            )

        self.logger.info(f"이미지 로드 시작: {input_source}")

        # URL 처리
        if input_source.startswith(("http://", "https://")):
            image = self._load_from_url(input_source)
        else:
            # 로컬 파일 처리
            image = self._load_from_file(input_source)

        self.logger.info(f"이미지 로드 완료: 크기 {image.shape}")
        return image

    def _load_from_url(self, url: str) -> np.ndarray:
        """
        URL에서 이미지 로드

        Args:
            url (str): 이미지 URL

        Returns:
            np.ndarray: 로드된 이미지 배열

        Raises:
            ValueError: URL에서 이미지 로드 실패 시
        """
        # URL에서 파일 확장자 추출
        url_path = url.split("?")[0]  # 쿼리 파라미터 제거
        extension = url_path.split(".")[-1].lower() if "." in url_path else "tmp"

        self.logger.info(f"URL에서 이미지 다운로드 중: {url}, 확장자: {extension}")

        try:
            with tempfile.NamedTemporaryFile(
                suffix=f".{extension}", delete=True
            ) as temp_file:
                response = requests.get(url, stream=True)
                response.raise_for_status()  # HTTP 에러 발생 시 예외 발생

                # 임시 파일에 저장
                with open(temp_file.name, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                return self._load_from_file(temp_file.name)
        except Exception as e:
            error_msg = f"URL에서 이미지 로드 실패: {url}\n{e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _load_from_file(self, file_path: str) -> np.ndarray:
        """
        로컬 파일에서 이미지 로드

        Args:
            file_path (str): 이미지 파일 경로

        Returns:
            np.ndarray: 로드된 이미지 배열

        Raises:
            ValueError: 파일 읽기 실패 또는 처리 오류 시
        """
        if not os.path.exists(file_path):
            error_msg = f"파일이 존재하지 않습니다: {file_path}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info(f"파일에서 이미지 읽기: {file_path}")

        try:
            # 이미지 읽기
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"이미지 파일을 읽을 수 없습니다: {file_path}")

            # 이미지 채널 처리
            return self._ensure_rgb(image)

        except Exception as e:
            error_msg = f"이미지 처리 오류: {file_path}\n{e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        이미지가 3채널 RGB 형식인지 확인하고, 필요하면 변환

        Args:
            image (np.ndarray): 원본 이미지 배열

        Returns:
            np.ndarray: 3채널 RGB 형식의 이미지 배열
        """
        # 그레이스케일 이미지
        if len(image.shape) == 2:
            self.logger.info("그레이스케일 이미지를 RGB로 변환")
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # RGBA 이미지 (4채널)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            self.logger.info("RGBA 이미지를 RGB로 변환")
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        # BGR 이미지 (3채널)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            self.logger.info("BGR 이미지를 RGB로 변환")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 기타 형식 (예상치 못한 채널 수)
        else:
            self.logger.warning(
                f"예상치 못한 이미지 형식: shape={image.shape}, 최선의 변환 시도"
            )
            try:
                # RGB로 변환 시도
                if len(image.shape) == 3:
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    return rgb
                else:
                    # 형식을 알 수 없는 경우, 먼저 그레이스케일로 변환 후 RGB로 변환
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                    return rgb
            except Exception as e:
                self.logger.error(f"이미지 변환 실패: {e}")
                raise ValueError(f"지원되지 않는 이미지 형식: shape={image.shape}")

    def __call__(self, input_source: str) -> np.ndarray:
        """
        클래스 인스턴스를 직접 호출할 수 있게 함 (편의 메서드)

        Args:
            input_source (str): 로컬 파일 경로 또는 URL

        Returns:
            np.ndarray: 로드된 RGB 이미지 배열
        """
        return self.load(input_source)
