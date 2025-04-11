from src.network.pipeline import DWPosePipeline
from src.utils.loader import ImageLoader
from src.utils.visualize import PoseVisualizer

REPO_ID = "Longcat2957/dwpose-onnx"

if __name__ == "__main__":
    imLoader = ImageLoader(verbose=False)
    image = imLoader.load("sample.png")

    pipeline = DWPosePipeline(
        device="cuda",
        verbose=False,
        det_repo_id=REPO_ID,
        det_filename="yolox_l.onnx",
        pose_repo_id=REPO_ID,
        pose_filename="dw-ll_ucoco_384.onnx",
    )

    out = pipeline(image)
    pv = PoseVisualizer(verbose=False, draw_feet=True)

    obj = pv(input_image=image, pose_data=out, format="pil")
    obj.save("output_pil.jpg")
