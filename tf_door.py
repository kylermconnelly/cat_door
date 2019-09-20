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
from keras import backend as be

import pygame

import board
import neopixel
import threading

import twitter

import moviepy.editor as mpy

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def read_tensor_from_image_file(img_arr, input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"

    # can manually set mean and std here
    #input_mean = 75
    #input_std = 54

    # convert numpy array to tensor
    #image_reader = tf.convert_to_tensor(img_arr, name=input_name)

    sess = tf.Session()
    
    float_caster = tf.cast(img_arr, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    normalized = tf.divide(tf.subtract(dims_expander, [input_mean]), [input_std])
    
    
    result = sess.run(normalized)
    sess.close()

    # each one of the tf.* operations above is adding data to some default graph
    # calling this clears that graph from memory
    be.clear_session()
    
    return result

def predict_from_image(img_sense, input_mean, input_std, graph):
    t = read_tensor_from_image_file(img_sense,
                                    input_mean=input_mean,
                                    input_std=input_std)
    
    sess = tf.Session(graph=graph)
    results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
    sess.close()

    del t
    
    results = np.squeeze(results)
    return results

# neo pixel animation
pix_pin = board.D21
pix_num = 8
pix_ord = neopixel.GRBW

pix = neopixel.NeoPixel(pix_pin, pix_num, brightness = 1.0,
                        auto_write = True, pixel_order = pix_ord)

def wheel(pos):
    # Input a value 0 to 255 to get a color value.
    # The colours are a transition r - g - b - back to r.
    if pos < 0 or pos > 255:
        r = g = b = 0
    elif pos < 85:
        r = int(pos * 3)
        g = int(255 - pos*3)
        b = 0
    elif pos < 170:
        pos -= 85
        r = int(255 - pos*3)
        g = 0
        b = int(pos*3)
    else:
        pos -= 170
        r = 0
        g = int(pos*3)
        b = int(255 - pos*3)
    return (r, g, b) if pix_ord == neopixel.RGB or pix_ord == neopixel.GRB else (r, g, b, 0)
 
 
def rainbow_cycle(wait):
    for j in range(255):
        for i in range(pix_num):
            pixel_index = (i * 256 // pix_num) + j
            pix[i] = wheel(pixel_index & 255)
        pix.show()
        time.sleep(wait)

global ledSt, idleLED, binglyLED, unknownLED, deterLED

idleLED = 0
binglyLED = 1
unknownLED = 2
deterLED = 3
ledSt = idleLED

exiting = False

def ledManager():
    global ledSt, idleLED, binglyLED, unknownLED, deterLED, exiting
    idxScale = 10
    idxMax = pix_num * idxScale
    idx = 0
    while True:
        if exiting == True:
            pix.fill((0, 0, 0, 0))
            break
        if ledSt == idleLED:
            if idx%idxScale == 0:
                pix.fill((0, 0, 0, 0))
                pix[int(idx/idxScale)] = (1, 0, 0, 0)
            
        elif ledSt == binglyLED:
            for i in range(pix_num):
                time.sleep(0.02)
                pix.fill((0, 0, 0, 0))
                pix[i] = (1, 0, 0, 0)
            ledSt = idleLED
            idx = idxMax
            
        elif ledSt == unknownLED:
            for i in range(pix_num):
                time.sleep(0.02)
                pix.fill((0, 0, 0, 0))
                pix[pix_num-i-1] = (1, 0, 0, 0)
            ledSt = idleLED
            idx = idxMax
            
        elif ledSt == deterLED:
            for i in range(4):
                pix.fill((0, 0, 0, 0))
                pix.fill((255, 0, 0, 0))
                time.sleep(0.1)
                pix.fill((0, 0, 0, 0))
                pix.fill((0, 255, 0, 0))
                time.sleep(0.1)
                pix.fill((0, 0, 0, 0))
                pix.fill((0, 0, 255, 0))
                time.sleep(0.1)
                pix.fill((0, 0, 0, 0))
                pix.fill((0, 0, 0, 255))
                time.sleep(0.1)
            
            #for i in range(4):
            #    pix.fill((0, 0, 0, 0))
            #    pix.fill((255, 0, 0, 0))
            #rainbow_cycle(0.001)
            #for i in range(pix_num):
            #    time.sleep(0.02)
            #    pix.fill((0, 0, 0, 0))
            #    pix[i] = (0, 0, 255, 0)
            ledSt = idleLED
            idx = idxMax
            
        idx += 1
        if idx >= idxMax:
            idx = 0
        time.sleep(0.1)

vs = 0
gettingImg = False
def getImage():
    global vs, gettingImg
    
    while gettingImg == True:
        time.sleep(0.01)
        
    gettingImg = True
    img =  vs.read()
    gettingImg = False
    return img
    

gif_list = []
num_gif_pics = 20
def gifManager():
    global exiting, gif_list, num_gif_pics

    while len(gif_list) < num_gif_pics:
        gif_list.append(getImage())
        
    while True:
        if exiting == True:
            break
        # add new image
        gif_list.append(getImage())
        # remove oldest
        gif_list.pop(0)
        time.sleep(0.1)

def main():
    global exiting, gif_list, vs, ledSt, idleLED, binglyLED, unknownLED, deterLED
    # start LED thread
    ledThread = threading.Thread(target = ledManager)
    ledThread.start()
    
    # setup logs and folders
    os.makedirs("Logs", exist_ok = True)
    log_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    os.makedirs(str("Logs/" + log_name), exist_ok = True)

    os.makedirs("gifs", exist_ok = True)

    width = 304
    height = 224

    # create camera instance
    vs = VideoStream(usePiCamera=True, resolution=(width, height)).start()

    # start GIF thread
    gifThread = threading.Thread(target = gifManager)
    gifThread.start()
    
    model_file = "2019_06_23_mobilenet_1_0_224.pb" #"tf_files/retrained_graph.pb"
    label_file = "2019_06_23_mobilenet_1_0_224_labels.txt" #"tf_files/retrained_labels.txt"
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    args = parser.parse_args()

    if args.graph:
        model_file = args.graph
    if args.labels:
        label_file = args.labels
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std

    graph = load_graph(model_file)
    labels = load_labels(label_file)

    # create a folder for each label to save pictures in
    for label in labels:
        os.makedirs(str("Logs/" + log_name + "/" + label), exist_ok = True)


    # declare images in memory
    np_img = np.empty((height, width, 3), dtype=np.uint8)
    # array of n arrays (which are sized to be images) to take moving average
    np_mv_ave = np.empty((10, height, width, 3), dtype=np.uint8)
    curr_move_ave = np.empty((height, width, 3), dtype=np.uint8)
    
    classCount = 0

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);
    
    # we use this for clipping images later
    bands = int((width - height)/2)  # how much to leave on each side

    # initialize audio player
    pygame.mixer.init()
    
    mv_ave_idx = 0
    init_mov_ave = False

    # init tf session
    sess = tf.Session(graph=graph)

    # create twitter object and authenticate
    f = open("twtrKeys.txt", "r")

    api = twitter.Api(consumer_key=str(f.readline().strip()),
                      consumer_secret=str(f.readline().strip()),
                      access_token_key=str(f.readline().strip()),
                      access_token_secret=str(f.readline().strip()))

    status = api.PostUpdate(datetime.now().strftime("%Y/%m/%d %H:%M:%S")+' Cat door is up!')
    print(status.text)
    timeSinceTweet = time.time()
    
    while True:
        np_img = getImage() #vs.read()

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
            
        if ((ave_delta > 2.0) and init_mov_ave) or False:
            # now see if motion is left, right, or centered
            left_delta = np.mean(abs_delta[:,:height])
            right_delta = np.mean(abs_delta[:,-height:])

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

            start = time.time()
            
            t = read_tensor_from_image_file(img_sense,
                                    input_mean=input_mean,
                                    input_std=input_std)

            results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
            
            results = np.squeeze(results)

            end = time.time()
            
            top_k = results.argsort()[-5:][::-1]

            print("\n"+datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
            print("Count: ", classCount)
            print("Ave Delta: ", str('%0.1f'%ave_delta))
            print('Evaluation time: {:.3f}s'.format(end-start) +
                  " Classification " + labels[top_k[0]])
            for i in range(len(top_k)):
                print(labels[i], '%0.2f'%results[i])
                
            # now save classified image in appropriate folder
            captureFileName = str("Logs/" + log_name + "/" + labels[top_k[0]] + '/'
                                  + labels[top_k[0]] + '_%2.1f%%'%(100*results[top_k[0]]) +
                                  str('_%0.1f_'%ave_delta) + str(classCount) + "_" + position +
                                  '_capture.jpg')
            scipy.misc.imsave(captureFileName, np_img)
            scipy.misc.imsave(str("Logs/" + log_name + "/" + labels[top_k[0]] + '/'
                                  + labels[top_k[0]] + '_%2.1f%%'%(100*results[top_k[0]]) +
                                  str('_%0.1f_'%ave_delta) + str(classCount) + "_" + position +
                                  '_delta.jpg'), abs_delta)
            classCount += 1
            
            # make noise for "pests"
            if (labels[top_k[0]] == "raccoon" or labels[top_k[0]] == "tchalla") and results[top_k[0]] > 0.9 or False:
                # let previous playback finish if it hasn't
                while pygame.mixer.music.get_busy() == True:
                    continue
                ledSt = deterLED
                pygame.mixer.music.load("dog_bark.wav")
                pygame.mixer.music.play()
            elif labels[top_k[0]] == "bingly":
                ledSt = binglyLED
            elif labels[top_k[0]] == "unknown":
                ledSt = unknownLED

            # tweet latest development if we are sure of result
            if results[top_k[0]] > 0.9 and labels[top_k[0]] != "unknown" or False:
                # only tweet every n seconds
                if time.time() > (timeSinceTweet + 10):
                    time.sleep(1.0) # sleep a bit so gif get before and after
                    fps = 11
                    temp = gif_list
                    clip = mpy.ImageSequenceClip(temp, fps=fps)
                    gif_fname = str('gifs/' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.gif')
                    clip.write_gif(gif_fname, fps=fps)
                    
                    timeSinceTweet = time.time()
                    try:
                        status = api.PostUpdate('%2.1f%% '%(100*results[top_k[0]]) + labels[top_k[0]]  +
                                                ' at door? ' + datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                                                media = gif_fname)
                        #captureFileName
                        print(status.text)
                    except:
                        print("Failed to tweet")
            
        time.sleep(0.2)
        # show image
        cv2.imshow("Live", np_img)
        # show delta image
        cv2.imshow("Delta", abs_delta)

        cv2.imshow("Mov Ave", curr_move_ave)
        
        # grab a key input
        key = cv2.waitKey(1) & 0xFF
        # check if we are to exit, clean exit
        if key == ord("q"):
            exiting = True
            print("Exiting program")
            break
        
    # cleen up everything
    cv2.destroyAllWindows()
    vs.stop()
    
if __name__ == "__main__":
    main()
