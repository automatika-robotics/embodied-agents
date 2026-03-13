from typing import Dict, List

import numpy as np


class LocalVLM:
    """Local VLM inference using Moondream2.

    :param model_path: Path to the model directory
    :param device: Device to run on ('cpu' or 'cuda')
    :param ncpu: Number of CPU threads
    """

    def __init__(self, model_path: str, device: str = "cuda", ncpu: int = 1):
        try:
            import moondream as md
            from PIL import Image
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Local VLM model requires PIL and the moondream package. "
                "Install it with: pip install pillow moondream"
            ) from e

        self.model = md.vl(model=model_path)
        self.pil_image = Image
        self.device = device
        self.ncpu = ncpu

    def __call__(self, inference_input: Dict) -> Dict:
        """Run VLM inference.

        :param inference_input: Dict with 'query' (messages list) and 'images' (list of np arrays)
        :returns: Dict with 'output' (str)
        """

        # Extract the text query from messages
        messages = inference_input["query"]
        query = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                query = msg["content"]
                break

        # Get images
        images: List[np.ndarray] = inference_input.get("images", [])
        if not images:
            return {"output": "No image provided."}

        # Convert numpy array to PIL Image (take first image)
        img = images[0]
        if isinstance(img, np.ndarray):
            # OpenCV BGR to RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = img[:, :, ::-1]
            pil_image = self.pil_image.fromarray(img)
        else:
            pil_image = img

        # Encode image and query
        encoded_image = self.model.encode_image(pil_image)
        answer = self.model.query(encoded_image, query)["answer"]

        return {"output": answer}
