from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import numpy as np
import tensorflow as tf

import argparse
import os
from datetime import datetime
import RPi.GPIO as GPIO
from PIL import Image
import picamera
import picamera.array
import scipy.misc
import cv2
import imutils
from imutils.video import VideoStream
from imutils.video import FPS

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def read_tensor_from_image_file(img_arr, input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"

    # convert numpy array to tensor
    image_reader = tf.convert_to_tensor(img_arr, name=input_name)
    
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    normalized = tf.divide(tf.subtract(dims_expander, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

if __name__ == "__main__":
    # setup logs and folders
    os.makedirs("Logs", exist_ok = True)
    log_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    os.makedirs(str("Logs/" + log_name), exist_ok = True)

    width = 304
    height = 224

    # create camera instance
    vs = VideoStream(usePiCamera=True, resolution=(width, height)).start()

    file_name = "tf_files/flower_photos/daisy/3475870145_685a19116d.jpg"
    model_file = "tf_files/retrained_graph.pb"
    label_file = "tf_files/retrained_labels.txt"
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    args = parser.parse_args()

    if args.graph:
        model_file = args.graph
    if args.image:
        file_name = args.image
    if args.labels:
        label_file = args.labels
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer

    graph = load_graph(model_file)
    labels = load_labels(label_file)

    # create a folder for each label to save pictures in
    for label in labels:
        os.makedirs(str("Logs/" + log_name + "/" + label), exist_ok = True)
    
    np_img = np.empty((height, width, 3), dtype=np.uint8)
    np_delta = np.empty((height, width, 3), dtype=np.uint8)

    # array of n arrays (which are sized to be images) to take moving average
    np_mv_ave = np.empty((10, height, width, 3), dtype=np.uint8)

    classCount = 0

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    # create opencv net for comparison
    #cvNet = cv2.dnn.readNetFromTensorflow(model_file, "tf_files/mobilenet_0.25_224/mobilenet_v1_0.25_224_eval.pbtxt")

    # we use this for clipping images later
    bands = int((width - height)/2)  # how much to leave on each side
    
    mv_ave_idx = 0
    init_mov_ave = False
    while True:
        np_delta = np_img
        np_img = vs.read()

        np_mv_ave[mv_ave_idx] = np_img
        curr_move_ave = np.uint8(np_mv_ave.mean(axis=0))
        
        mv_ave_idx += 1

        if mv_ave_idx >= 10:
            mv_ave_idx = 0
            init_mov_ave = True

        # look for changes in image before processing
        abs_delta = np.abs(np.subtract(rgb2gray(np_img), rgb2gray(curr_move_ave)))
        abs_delta[abs_delta < 10] = 0
        ave_delta = np.mean(abs_delta)
            
        if (ave_delta > 0.1) and init_mov_ave and True:
            print("Ave Delta: ", str(ave_delta))

            # now see if motion is left, right, or centered
            left_delta = np.mean(abs_delta[:,:height])
            right_delta = np.mean(abs_delta[:,-height:])

            print("Left Delta " + str(left_delta) +
                      ", Right Delta " + str(right_delta))

            # now that we have average motion, choose side accordingly
            if left_delta > (right_delta + 0.02):
                img_sense = np_img[:,:height,:]
                position = "Left"
            elif right_delta > (left_delta + 0.02):
                img_sense = np_img[:,-height:,:]
                position = "Right"
            else:
                # not a strong bias towards either side, choose middle
                img_sense = np_img[:,bands:-bands,:]
                position = "Center"
            
            t = read_tensor_from_image_file(img_sense,
                                            input_mean=input_mean,
                                            input_std=input_std)
                                
            with tf.Session(graph=graph) as sess:
                start = time.time()
                results = sess.run(output_operation.outputs[0],
                                   {input_operation.outputs[0]: t})
                end=time.time()
                
            results = np.squeeze(results)
            top_k = results.argsort()[-5:][::-1]

            print("Count: ", classCount)
            print('Evaluation time: {:.3f}s'.format(end-start))
            for i in top_k:
                print(labels[i], results[i])
                
            # now save classified image in appropriate folder
            scipy.misc.imsave(str("Logs/" + log_name + "/" + labels[top_k[0]] + '/'
                                  + labels[top_k[0]] + '_%2.1f%%'%(100*results[top_k[0]]) + '_' +
                                  str(classCount) + "_" + position + '_capture.jpg'), np_img)
            scipy.misc.imsave(str("Logs/" + log_name + "/" + labels[top_k[0]] + '/'
                                  + labels[top_k[0]] + '_%2.1f%%'%(100*results[top_k[0]]) + '_' +
                                  str(classCount) + "_" + position + '_delta.jpg'), abs_delta)
            classCount += 1
            
        time.sleep(1.0)
        # show image
        cv2.imshow("Live", np_img)
        # show delta image
        cv2.imshow("Delta", abs_delta)

        cv2.imshow("Mov Ave", curr_move_ave)
        
        # grab a key input
        key = cv2.waitKey(1) & 0xFF
        # check if we are to exit, clean exit
        if key == ord("q"):
            print("Exiting program")
            break
        
    # cleen up everything
    cv2.destroyAllWindows()
    vs.stop()