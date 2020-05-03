#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.plugin = None
        self.net = None

    def load_model(self, model_xml, device_name="CPU"):
        # Loading the model
        self.plugin = IECore()
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.net = IENetwork(model=model_xml, weights=model_bin)

        # Check for supported layers and Unsupported layers
        # If Unsupported layers found then exit the program
        supported_layers = self.plugin.query_network(network=self.net, device_name=device_name)
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Following layers are not supported: {}".format(unsupported_layers))
            exit(1)
        
        # Load the network to IR and return 
        ir_model = self.plugin.load_network(self.net, device_name)
        return ir_model

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return

    def exec_net(self):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return
