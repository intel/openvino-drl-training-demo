from openvino.inference_engine import IECore
import torch.nn as nn
import torchvision
from torchvision import  models, transforms
from PIL import Image
import torch

num_classes = 2
input_size = 224

def pre_process(img):

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

def init_vino_model(file, device='CPU'):

    model_xml = file + ".xml"
    model_bin = file + ".bin"
    model_onnx = file + ".onnx"

    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    exec_net = ie.load_network(network=net, device_name=device)
    return exec_net

def vino_inference(image, network):
    input = pre_process(image)
    input_blob = next(iter(network.input_info))
    out_blob = next(iter(network.outputs))
    res = network.infer(inputs={input_blob: input})
    output = torch.tensor(res[out_blob])
    final = post_process(output)
    return final
