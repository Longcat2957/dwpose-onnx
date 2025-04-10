# 🧍‍♂️ DWPose-ONNX

**DWPose with ONNX Runtime (GPU Supported)**  
본 저장소는 YOLOX 기반 객체 탐지기와 DWPose 기반 포즈 추정기를 ONNX로 구성한 경량 추론 파이프라인입니다. 효율성과 실용성을 모두 고려하여 설계되었습니다.

---

## 🚀 주요 특징

- 🔧 **단일 추론 파이프라인**  
  `DWPosePipeline` 클래스를 통해 객체 탐지 및 포즈 추정을 일관된 흐름으로 처리합니다.

- 🏃 **134개 전신 키포인트 추정**  
  신체, 얼굴, 손, 발을 포함한 전신 키포인트 지원.

- 📦 **ONNX 경량화 모델**  
  CPU 및 GPU 환경에서 손쉽게 실행 가능.

- 🧩 **Hugging Face Hub 통합**  
  모델 자동 다운로드 및 배포 경로 지정 지원.

- 🖼️ **직관적인 시각화 도구 제공**  
  `PoseVisualizer`를 통해 추정 결과를 바로 시각화할 수 있습니다.

---

## 🛠️ 설치 방법

```bash
git clone https://github.com/Longcat2957/dwpose-onnx.git
cd dwpose-onnx
pip install -r requirements.txt
```

---

## 🔧 사용 예시

```python
from network.pipeline import DWPosePipeline
from utils.loader import ImageLoader
from utils.visualize import PoseVisualizer

REPO_ID = "Longcat2957/dwpose-onnx"

imLoader = ImageLoader()
image = imLoader.load("sample.png")

pipeline = DWPosePipeline(
    device="cuda",  # 또는 "cpu"
    det_repo_id=REPO_ID,
    det_filename="yolox_l.onnx",
    pose_repo_id=REPO_ID,
    pose_filename="dw-ll_ucoco_384.onnx",
)

pose_output = pipeline(image)

pv = PoseVisualizer()
result = pv(image, pose_data=pose_output, format="pil")
result.save("output.jpg")
```

---

## 📂 프로젝트 구조

```
dwpose-onnx/
├── main.py                   # 실행 예제
├── network/                  # 탐지기 및 포즈 추정기 구성
├── utils/                    # 이미지 로더 및 시각화 유틸리티
├── sample.png                # 샘플 이미지
├── requirements.txt
└── README.md
```

---

## 🧠 키포인트 구성

| 부위     | 개수         |
|----------|--------------|
| 신체     | 18개         |
| 발       | 6개 (좌/우 각 3개) |
| 얼굴     | 68개         |
| 손       | 42개 (좌/우 각 21개) |
| **총합** | **134개**    |

---

## 🗂️ 모델 정보

| 모델 종류 | 파일명                   | 설명                          |
|----------|--------------------------|-------------------------------|
| Detector | `yolox_l.onnx`           | 사람 객체 탐지를 위한 YOLOX |
| Pose     | `dw-ll_ucoco_384.onnx`   | 133개 키포인트 예측 DWPose  |

> 최초 실행 시 Hugging Face Hub에서 자동으로 모델이 다운로드됩니다.  
> 별도로 모델을 배포하고자 할 경우, `repo_id` 및 `filename` 파라미터를 지정하십시오.

---

## ⚠️ 참고 사항

- `DWPosePipeline`은 OpenPose 포맷 호환을 위해 목(neck) 키포인트를 보간하여 포함합니다.
- 추론 결과는 `body`, `face`, `hands`, `feet`으로 자동 분할됩니다.
- 각 키포인트는 `(x, y, score)` 형식으로 출력되며, 입력 이미지 크기를 기준으로 정규화됩니다.

---

## 🤝 기여 안내

본 프로젝트는 모든 개발자와 연구자를 위한 공개 프로젝트입니다.  
이슈 제기 및 기능 제안, 코드 기여는 언제든지 환영합니다.

1. 이슈 또는 Pull Request를 생성해 주세요.
2. 변경 사항은 명확하고 간결하게 설명해 주세요.
3. 문서 및 주석을 함께 작성해 주시면 감사하겠습니다.

> 본 프로젝트가 도움이 되셨다면 ⭐️ Star를 통해 응원해 주세요.

---

