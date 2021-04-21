from openvino.inference_engine import IECore
from .reward_inference import pre_process, post_process
import torch

def init_vino_model(file, device='CPU'):

    model_onnx = file + ".onnx"
    ie = IECore()
    net = ie.read_network(model=model_onnx)
    exec_net = ie.load_network(network=net, device_name=device)
    return exec_net

def vino_inference(network, image):
    input = pre_process(image)
    input_blob = next(iter(network.input_info))
    out_blob = next(iter(network.outputs))
    res = network.infer(inputs={input_blob: input})
    output = torch.tensor(res[out_blob])
    return post_process(output)
