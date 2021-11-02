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
