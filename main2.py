from network.pipeline import DWPosePipeline
from utils.loader import ImageLoader

REPO_ID = "Longcat2957/dwpose-onnx"

if __name__ == "__main__":
    imLoader = ImageLoader(verbose=True)
    image = imLoader.load("7.jpg")

    pipeline = DWPosePipeline(
        device="cuda",
        verbose=True,
        det_repo_id=REPO_ID,
        det_filename="yolox_l.onnx",
        pose_repo_id=REPO_ID,
        pose_filename="dw-ll_ucoco_384.onnx",
    )

    out = pipeline(image)