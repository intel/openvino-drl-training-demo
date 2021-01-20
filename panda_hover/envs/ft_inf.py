import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image

def initialize_model(model_name, file, num_classes=2):

    if model_name == 'squeezenet':
        model = models.squeezenet1_0()
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))

    elif model_name == 'vgg':
        model = models.vgg11_bn()
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet":
        model = models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    model.num_classes = num_classes
    model.load_state_dict(torch.load(file))
    model.eval()
    return model

def pre_process(img):

    input_size = 224

    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
    }

    np_array = img[:, :, ::-1]
    image = Image.fromarray(np_array, 'RGB')

    # transform the image
    im_tr = data_transforms['val'](image)
    return im_tr.unsqueeze(0)

def post_process(outputs):

    _, preds = torch.max(outputs, 1)

    probabilities= torch.nn.functional.softmax(outputs[0], dim=0)

    probs = probabilities.detach().numpy()

    return -100.0 + probs[0]*100.0

def inference(img, model):
    input = pre_process(img)
    output = model(input)
    final = post_process(output)
    return final
