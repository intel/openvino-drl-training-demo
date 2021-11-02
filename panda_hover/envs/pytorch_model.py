from .inference_utils import pre_process, post_process
from torchvision import models
import torch.nn as nn
import torch

class PyTorchModel:
    """
     Class that creats an inference ready Squeezenet1_0 Pytorch Model from a file
    """
    _model = None

    def __init__(self, filename):
        """
            input
                - filename of pytorch model
        """
        self._model = models.squeezenet1_0()
        self._model.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
        self._model.num_classes = 2
        self._model = torch.load(filename)
        self._model.eval()

    def inference(self, image):
        """
            Runs inference on an image

            input
                - image
        """
        processed_image = pre_process(image)
        output = self._model(processed_image)
        return post_process(output)
