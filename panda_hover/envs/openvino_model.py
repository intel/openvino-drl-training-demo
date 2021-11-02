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
