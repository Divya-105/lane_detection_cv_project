"""
Lane Lines Detection pipeline

Usage:
    main.py [--video] INPUT_PATH OUTPUT_PATH 

Options:

-h --help                               show this screen
--video                                 process video file instead of image
"""

import os
import tempfile
import cv2
import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import sys

import numpy as np
import matplotlib.image as mpimg
import cv2
from docopt import docopt
from IPython.display import HTML, Video
from moviepy.editor import VideoFileClip
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *

class FindLaneLines:
    """ This class is for parameter tunning.

    Attributes:
        ...
    """
    def __init__(self):
        """ Init Application"""
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, img):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        return out_img

    def process_image(self, input_path, output_path):
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        mpimg.imsave(output_path, out_img)

    def process_video(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile(output_path, audio=False)

def main():
    # findLaneLines = FindLaneLines()
    # findLaneLines.process_video("harder_challenge_video.mp4", "output.mp4")
    findLaneLines = FindLaneLines()

    st.title('Lane Detection')
    st.header('Upload a video and detect lane')

    st.header('Input Video')

    f = st.file_uploader("Upload file")
    if f is not None:
        file_details = {"FileName":f.name,"FileType":f.type}
        st.write(file_details)
        # st.write(type(st.video_file))
        # vid = st.load_video(f)
    # with open(os.path.join("Streamlit - Copy",f.name),"wb") as f: 
    #       f.write(f.getbuffer())         
    with open(f.name,"wb") as ff: 
        ff.write(f.getbuffer())         
    st.success("Saved File")
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(f.read())

    vf = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    # while vf.isOpened():
    #     ret, frame = vf.read()
    #     # if frame is read correctly ret is True
    #     if not ret:
    #         print("Can't receive frame (stream end?). Exiting ...")
    #         break
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     stframe.image(gray)
        
    st.header('Processing......')


    white_output = './output.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    #clip1 = VideoFileClip("test_videos/jaipurHighway.mp4").subclip(50,60)
    # os.path.join("tempDir",f.name),"wb"

    clip1 = VideoFileClip(f.name)#("input.mp4")
    
    # white_clip = clip1.fl_image(findLaneLines.process_video(f.name,"output.mp4")) #NOTE: this function expects color images!!
    findLaneLines.process_video(f.name,"output.mp4")
    # white_clip.write_videofile(white_output, audio=False)
    st.write("hii")
    video_file = open('output.mp4', 'rb')
    st.write("hello")
    video_bytes = video_file.read()
    st.write("heyyy")
    st.video(video_bytes)

    
    

if __name__ == "__main__":
    main()
