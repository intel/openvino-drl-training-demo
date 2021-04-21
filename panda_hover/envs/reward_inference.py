import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def init_model(model, filename):
    if model == 'squeezenet':
        model = models.squeezenet1_0()
        model.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = 2

    elif model == 'resnet':
        model = models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    model = torch.load(filename)
    return model.eval()

def pre_process(image):
    image = image[:, :, ::-1]
    image = Image.fromarray(image)
    transformed_image = data_transforms(image)
    return transformed_image.unsqueeze(0)

def post_process(output):
    _, preds = torch.max(output, 1)
    probabilities= torch.nn.functional.softmax(output[0], dim=0)
    return preds.detach().numpy()[0], probabilities

def inference(model, image):
    processed_image = pre_process(image)
    output = model(processed_image)
    return post_process(output)
