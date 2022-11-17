import tempfile
import cv2
import streamlit as st
import pandas as pd
import numpy as np
# import numpy as np
# import matplotlib.image as mpimg
# import cv2
# from docopt import docopt
# from IPython.display import HTML
# from IPython.core.display import Video
# from moviepy.editor import VideoFileClip
# from CameraCalibration import CameraCalibration
# from Thresholding import *
# from PerspectiveTransformation import *
# from LaneLines import *

# from copy_of_lane_detection import main

st.title('Lane Detection')
st.markdown('Upload a video and detect lane')

st.header('Input Video')
# st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

# video_file = open('harder_challenge_video.mp4', 'rb')
# video_bytes = video_file.read()
# st.video(video_bytes)


# col1, col2 = st.columns(2)
# with col1:
#     st.text('Sepal characteristics')
#     sepal_l = st.slider('Sepal lenght (cm)', 1.0, 8.0, 0.5)
#     sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)
# with col2:
#     st.text('Pepal characteristics')
#     petal_l = st.slider('Petal lenght (cm)', 1.0, 7.0, 0.5)
#     petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)

# st.button('Predict type of Iris')



# class FindLaneLines:
#     def __init__(self):
#         """ Init Application"""
#         self.calibration = CameraCalibration('camera_cal', 9, 6)
#         self.thresholding = Thresholding()
#         self.transform = PerspectiveTransformation()
#         self.lanelines = LaneLines()

#     def forward(self, img):
#         out_img = np.copy(img)
#         img = self.calibration.undistort(img)
#         img = self.transform.forward(img)
#         img = self.thresholding.forward(img)
#         img = self.lanelines.forward(img)
#         img = self.transform.backward(img)

#         out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
#         out_img = self.lanelines.plot(out_img)
#         return out_img

#     def process_image(self, input_path, output_path):
#         img = mpimg.imread(input_path)
#         out_img = self.forward(img)
#         mpimg.imsave(output_path, out_img)

#     def process_video(self, input_path, output_path):
#         clip = VideoFileClip(input_path)
#         out_clip = clip.fl_image(self.forward)
#         out_clip.write_videofile(output_path, audio=False)

# def main():
f = st.file_uploader("Upload file")

tfile = tempfile.NamedTemporaryFile(delete=False) 
tfile.write(f.read())

vf = cv2.VideoCapture(tfile.name)

stframe = st.empty()

while vf.isOpened():
    ret, frame = vf.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stframe.image(gray)
#     findLaneLines = FindLaneLines()
#     findLaneLines.process_video(f,"output.mp4")
#     # return "output.mp4";
    
#     # f = st.file_uploader("Upload file")
#     clip1 = VideoFileClip("output.mp4")
#     tfile = tempfile.NamedTemporaryFile(delete=False) 
#     tfile.write(clip1.read())
#     vf = cv2.VideoCapture(tfile.name)
#     stframe = st.empty()

# if __name__ == "__main__":
#     main()
# clip= st.video(output, format: str = "video/mp4", start_time: int = 0)
# cv.imshow("output.mp4", output.mp4)
    # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
# if cv2.waitKey(10) & 0xFF == ord('q'):
#     break

    

video_file = open('output.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)