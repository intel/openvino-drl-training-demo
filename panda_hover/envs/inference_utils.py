# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from torchvision import transforms
from PIL import Image
import torch

data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def pre_process(image):
    """
    Pre-processes image based on TorchVision Transforms before inference

    input
        - image in bgr format

    output
        - pre-processed image

    """
    image = image[:, :, ::-1] #change to rgb format
    image = Image.fromarray(image)
    transformed_image = data_transforms(image)
    return transformed_image.unsqueeze(0)

def post_process(output):
    """
    Returns class prediciton based on the inference output

    input
        - output of a Neural Network inference

    output
        - class with highest probability
    """
    _, preds = torch.max(output, 1)
    return preds.detach().numpy()[0]
