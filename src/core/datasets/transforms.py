from typing import Any
import cv2
import numpy as np
from PIL import Image

class Compose:
    """
    Do transformation on input data with corresponding pre-processing and augmentation operations.
    The shape of input data to all operations is [height, width, channels].

    Args:
        transforms (list): A list contains data pre-processing or augmentation. An empty list means only reading images, no transformation.
        to_rgb (bool, optional): If converting image to RGB color space. Default: True.
        img_channels (int, optional): The image channels used to check the loaded image. Default: 3.

    Raises:
        TypeError: When "transforms" is not a list.
        ValueError: when the length of "transforms" is less than 1.
    """

    def __init__(self, 
                 transforms,
                 mode,
                 input_data_format_postfix: str = "npy",
                 output_data_format_postfix: str = "npy"):
        if not isinstance(transforms, list):
            raise TypeError("The transforms must be a list!")
        self.transforms = transforms
        self.mode = mode

    def __call__(self, data):
        """
        Args:
            data: A dict to deal with. It may include keys: "img", "label", "trans_info" and "gt_fields".
                "trans_info" reserve the image shape informating. And the "gt_fields" save the key need to transforms
                together with "img"

        Returns: A dict after process。
        """
        
        for op in self.transforms:
            data = op(data, self.mode)

        return data

class Label_Normalization:

    def __init__(self):
        self.min_value = 0
        self.max_value = 0
    
    def __call__(self, data, mode) -> Any:
        if mode == "train":
            self.min_value = np.min(data["label"])
            self.max_value = np.max(data["label"])
        else:
            if (self.min_value == 0) and (self.max_value == 0):
                raise ValueError("The min_value and max_value should be set in training mode.")

        data["label"] = (data["label"] - self.min_value) / (self.max_value - self.min_value)
        return data

class Add_Dimension:

    def __init__(self, axis: int = 1):
        self.axis = axis

    def __call__(self, data, mode) -> Any:
        if data["img"].ndim == 3:
            data["img"] = np.expand_dims(data["img"], axis=self.axis)
        if data["label"].ndim == 3:
            data["label"] = np.expand_dims(data["label"], axis=self.axis)
        return data