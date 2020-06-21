"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
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
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
import utils

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    # Connect to the MQTT client
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    # Load the model through `infer_network`
    model = infer_network.load_model(args.model, args.cpu_extension, args.device)
    input_shape = infer_network.get_input_shape()
    # Handle the input stream
    # Identify if image, video or camera and process accordingly
    input_type = utils.get_file_type(args.input)

    cap = None
    im_flag = False
    total_count = 0
    last_count = 0
    start_time = 0

    if input_type == "IMAGE":
        im_flag = True
    elif input_type == "VIDEO":
        cap = cv2.VideoCapture(args.input)
    elif input_type == "CAM":
        cap = cv2.VideoCapture(0)
    else:
        print("Unrecognized image or video file format. Please provide proper path or 'CAM' for camera input")
        sys.exit(1)
    
    cap.open(args.input)

    # Loop until stream is over
    while cap.isOpened():
        # Read from the video capture
        flag, frame = cap.read()

        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        if key_pressed == 27:
            break

        width = int(cap.get(3))
        height = int(cap.get(4))

        # Pre-process the image as needed
        p_frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        # Start asynchronous inference for specified request
        infer_network.exec_net(model, p_frame)

        # Wait for the result
        if infer_network.wait() == 0:
            # Get the results of the inference request
            result = infer_network.get_output()

            # Extract any desired stats from the results
            out_frame, current_count = get_stats_draw_box(frame, result, width, height, args.prob_threshold)

            # Calculate and send relevant information on
            # current_count, total_count and duration to the MQTT server
            # Topic "person": keys of "count" and "total"
            # Topic "person/duration": key of "duration"
            if current_count > last_count:
                start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))
            if current_count < last_count:
                duration = int(time.time() - start_time)
                client.publish("person/duration", json.dumps({"duration": duration}))
            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count
        # Send the frame to the FFMPEG server
        sys.stdout.buffer.write(out_frame)
        sys.stdout.flush()
        
        # Write an output image if `single_image_mode`
        if im_flag:
            cv2.imwrite("test_out.jpg", out_frame)
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

def get_stats_draw_box(image, result, width, height, threshold):
    count = 0
    for box in result[0][0]:
        confidence = box[2]
        if confidence >= threshold:
            count += 1
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 255), 1)
            
    return image, count

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
