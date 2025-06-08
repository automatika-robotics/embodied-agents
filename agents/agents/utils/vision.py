import cv2
from typing import Dict
import numpy as np

try:
    import onnxruntime as ort
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "enable_local_classifier in Vision component requires onnxruntime to be installed. Please install them with `pip install onnxruntime` or `pip install onnxruntime-gpu` for cpu or gpu based deployment."
    ) from e


class LocalVisionModel:
    """Implements inference for a fast local detection model. The default model selected model is:
          @misc{huang2024deim,
          title={DEIM: DETR with Improved Matching for Fast Convergence},
          author={Shihua Huang, Zhichao Lu, Xiaodong Cun, Yongjun Yu, Xiao Zhou, and Xi Shen},
          booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
          year={2025},
    }
    """

    def __init__(
        self,
        model_path: str,
        ncpu: int = 1,
        device: str = "cpu",
    ):
        # Initialize the ONNX model
        sessionOptions = ort.SessionOptions()
        sessionOptions.inter_op_num_threads = ncpu
        sessionOptions.intra_op_num_threads = ncpu

        if (
            device == "gpu"
            and "CUDAExecutionProvider" not in ort.get_available_providers()
        ):
            import logging

            logging.getLogger("local_classifier").warning(
                "CUDAExecutionProvider is not available for local_classifier, ensure you have the correct CUDA and cuDNN versions installed and install onnx runtime with `pip install onnxruntime-gpu`. Switching to CPU runtime."
            )
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.model = ort.InferenceSession(
            model_path, sess_options=sessionOptions, providers=providers
        )

    def __resize_with_aspect_ratio(
        self, image, height, width, interpolation=cv2.INTER_LINEAR
    ):
        """Resizes an image while maintaining aspect ratio and pads it."""
        original_height, original_width = image.shape[:2]
        ratio = min(width / original_width, height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        # Resize the image
        resized_image = cv2.resize(
            image, (new_width, new_height), interpolation=interpolation
        )

        # Create a new image with the desired size and paste the resized image onto it
        new_image = np.full((height, width, 3), 128, np.uint8)  # Create a gray canvas
        pad_top = (height - new_height) // 2
        pad_left = (width - new_width) // 2
        new_image[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = (
            resized_image
        )

        im_t = np.transpose(new_image, (2, 0, 1))  # HWC to CHW

        return im_t

    def __call__(
        self,
        inference_input: Dict,
        img_height: int,
        img_width: int,
        dataset_labels: Dict,
    ) -> Dict:
        """
        Inference for vision model
        """
        # Create the size array using NumPy, matching the expected int64 dtype
        orig_size_np = np.array([[img_height, img_width]], dtype=np.int64)

        # Convert image to tensor-like numpy array
        im_data_np = (
            np.array(
                [
                    self.__resize_with_aspect_ratio(
                        img, img_height, img_width
                    )  # preprocess
                    for img in inference_input["images"]
                ],
                dtype=np.float32,
            )
            / 255  # Normal to Normalize to [0, 1]
        )

        results = []

        try:
            detection = self.model.run(
                output_names=None,
                input_feed={"images": im_data_np, "orig_target_sizes": orig_size_np},
            )

            # format results
            labels, boxes, scores = detection
            result = {}
            if boxes.size == 0:
                scores = detection.pred_instances.scores.cpu().numpy()
                labels = detection.pred_instances.labels.cpu().numpy()
                bboxes = detection.pred_instances.bboxes.cpu().numpy()
                # filter for threshold
                mask = scores >= inference_input["threshold"]
                scores, labels, bboxes = scores[mask], labels[mask], bboxes[mask]
                # Check if predictions survived thresholding
                if not (scores.size == 0):
                    # if labels are requested in text
                    if inference_input["get_dataset_labels"]:
                        # get text labels from model dataset info
                        labels = np.vectorize(
                            lambda x: dataset_labels[x],
                        )(labels)

                result = {
                    "bboxes": bboxes.tolist(),
                    "labels": labels.tolist(),
                    "scores": scores.tolist(),
                }

            if result:
                results.append(result)

        except Exception as e:
            import logging

            logging.getLogger("local_classifier").error(e)

        return {"output": results}


_MS_COCO_LABELS = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}
