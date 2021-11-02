from .inference_utils import pre_process, post_process
from openvino.inference_engine import IECore
import torch

class OpenVinoModel:
    """
    Class that creats an exectuable OpenVino Model from .onnx file"
    """
    _exec_net = None

    def __init__(self, filename, device='CPU'):
        """
            input
                - filename of .onnx file of model
                - device to run inference on
        """
        ie = IECore()
        net = ie.read_network(model=filename+".onnx")
        self._exec_net = ie.load_network(network=net, device_name=device)

    def inference(self, image):
        """
            Runs inference on an image

            input
                - image
        """
        input = pre_process(image)
        input_blob = next(iter(self._exec_net.input_info))
        out_blob = next(iter(self._exec_net.outputs))
        res = self._exec_net.infer(inputs={input_blob: input})
        output = torch.tensor(res[out_blob])
        return post_process(output)
