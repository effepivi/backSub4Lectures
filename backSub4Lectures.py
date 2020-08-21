#!/usr/bin/env python3

from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import copy

def nothing(x):
    pass

def clickAndCrop(event, x, y, flags, param):
    global ROI_corners, cropping, frame, update_ROI;

    if event == cv.EVENT_LBUTTONDOWN:
        ROI_corners = [(x, y)];
        cropping = True;
    elif cropping == True:
        if event == cv.EVENT_LBUTTONUP or event == cv.EVENT_MOUSEMOVE:
            if len(ROI_corners) == 1:
                ROI_corners.append((x, y));
            elif len(ROI_corners) == 2:
                ROI_corners[1] = (x, y);
        if event == cv.EVENT_LBUTTONUP:
            cropping = False;
            update_ROI = True;


parser = argparse.ArgumentParser(description='Background subtraction to use with Chroma key.')
parser.add_argument('--ROI', type=int, help='Corners of the region of interests.', nargs=4, required=False);
parser.add_argument('--colour', type=int, help='Replace the backbround with this RGB colour (0 0 0 is black, 255 255 255 is white), useful for chroma key.', nargs=3, required=False, default=[0, 255, 0]);
parser.add_argument('--threshold', type=int, help='Threshold.', nargs=1, required=False, default=64);
args = parser.parse_args()





capture = cv.VideoCapture(0)
if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)

NoneType = type(None);
background = None;
frame = None;
tot_frame = 10;
frame_set = [];
next_frame_id = 0;

end_loop = False;

kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3));
kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3));
background_colour_image = None;
next_frame_id = 0;

def addFrame(capture):
    global frame, next_frame_id;

    ret, temp = capture.read()

    temp = temp.astype("float");
    if len(frame_set) < tot_frame:
        next_frame_id += 1;
        frame_set.append(temp);
    else:
        frame_set[next_frame_id] = temp;
        next_frame_id += 1;

    if next_frame_id == tot_frame:
        next_frame_id = 0;

    for i in range(len(frame_set)):
            if i == 0:
                frame = temp;
            else:
                frame = np.add(frame, temp);

    frame = np.divide(frame, len(frame_set));
    frame = frame.astype('uint8');



ROI_corners = [];
cropping = False;
update_ROI = False;
ROI_background = None;

background_colour = args.colour;

threshold = 64;

if not isinstance(args.ROI, NoneType):
    ROI_corners = [(min(args.ROI[0], args.ROI[2]), min(args.ROI[1], args.ROI[3])), (max(args.ROI[0], args.ROI[2]), max(args.ROI[1], args.ROI[3]))];
    cropping = False;
    update_ROI = True;

if isinstance(args.threshold, int):
    threshold = args.threshold;
else:
    threshold = args.threshold[0];


while not end_loop:

    addFrame(capture);

    if isinstance(frame, NoneType):
        end_loop = True;
    else:

        if isinstance(background_colour_image, NoneType):
            cv.namedWindow("frame", cv.WINDOW_AUTOSIZE);
            cv.setMouseCallback("frame", clickAndCrop);
            cv.createTrackbar('Threshlod', 'frame', threshold, 255, nothing);
            cv.createTrackbar('R', 'frame', background_colour[0], 255, nothing);
            cv.createTrackbar('G', 'frame', background_colour[1], 255, nothing);
            cv.createTrackbar('B', 'frame', background_colour[2], 255, nothing);

            background_colour_image = np.zeros((frame.shape[0], frame.shape[1], 3), np.float);
            background_colour_image[:] = (background_colour[2] / 255, background_colour[1] / 255, background_colour[0] / 255);

        if len(ROI_corners) == 2:
            ROI_frame = copy.deepcopy(frame[ROI_corners[0][1]:ROI_corners[1][1], ROI_corners[0][0]:ROI_corners[1][0]]);
        else:
            ROI_frame = frame;

        if not isinstance(background, NoneType):
            if isinstance(ROI_background, NoneType):
                ROI_background = background;

            if cropping == False and ROI_frame.shape[0] > 0 and ROI_frame.shape[1] > 0:
                if update_ROI or ROI_background.shape[0] != ROI_frame.shape[0] or ROI_background.shape[1] != ROI_frame.shape[1]:
                    ROI_background = copy.deepcopy(background[ROI_corners[0][1]:ROI_corners[1][1], ROI_corners[0][0]:ROI_corners[1][0]]);

                    update_ROI = False;

                    background_colour_image = np.zeros((ROI_frame.shape[0], ROI_frame.shape[1], 3), np.float);
                    background_colour_image[:] = (background_colour[2] / 255, background_colour[1] / 255, background_colour[0] / 255);


                if background_colour[0] / 255 != cv.getTrackbarPos('R','frame') / 255 or background_colour[1] / 255 != cv.getTrackbarPos('G','frame') / 255 or background_colour[2] / 255 != cv.getTrackbarPos('B','frame') / 255:

                    background_colour[0] = cv.getTrackbarPos('R','frame');
                    background_colour[1] = cv.getTrackbarPos('G','frame');
                    background_colour[2] = cv.getTrackbarPos('B','frame');

                    background_colour_image = np.zeros((ROI_frame.shape[0], ROI_frame.shape[1], 3), np.float);
                    background_colour_image[:] = (background_colour[2] / 255, background_colour[1] / 255, background_colour[0] / 255);

                if ROI_background.shape[0] == ROI_frame.shape[0] and ROI_background.shape[1] == ROI_frame.shape[1]:
                    print(3)

                    diff = cv.absdiff(cv.pyrDown(cv.pyrDown(ROI_frame)), cv.pyrDown(cv.pyrDown(ROI_background)));

                    grey = cv.normalize(diff, np.zeros(diff.shape), 0, 255, cv.NORM_MINMAX);
                    grey = cv.cvtColor(grey, cv.COLOR_BGR2GRAY);


                    ret, foreground_mask = cv.threshold(grey, cv.getTrackbarPos('Threshlod','frame'), 255, cv.THRESH_BINARY);

                    foreground_mask = cv.medianBlur(foreground_mask, 1);
                    foreground_mask = cv.dilate(foreground_mask, kernel2, iterations = 3);
                    foreground_mask = cv.morphologyEx(foreground_mask, cv.MORPH_CLOSE, kernel1);
                    foreground_mask = cv.morphologyEx(foreground_mask, cv.MORPH_OPEN, kernel2);
                    foreground_mask = foreground_mask.astype('float') / 255;


                    alpha3 = cv.pyrUp(cv.pyrUp(np.stack([foreground_mask]*3, axis=2)));
                    alpha3 = alpha3[0:ROI_frame.shape[0], 0:ROI_frame.shape[1]];

                    cv.imshow('chroma key', alpha3 * ROI_frame / (255) + (1-alpha3) * background_colour_image);

        keyboard = cv.waitKey(30)
        if keyboard == 'b' or keyboard == ord('b') or keyboard == 'r' or keyboard == ord('r'):
            print("Get background");

            for i in range(tot_frame):
                print(i, '/', tot_frame);
                ret, temp = capture.read();
                temp = temp.astype("float")
                if i == 0:
                    background = temp;
                else:
                    background = np.add(background, temp);

            background = np.divide(background, tot_frame);
            background = background.astype('uint8');

            ROI_background = cv.GaussianBlur(background, (5,5),0);

            if len(ROI_corners) == 2:
                update_ROI = True;

        elif keyboard == 'q' or keyboard == ord('q') or keyboard == 27:
            end_loop = True;
        elif keyboard != -1:
            print("unknown key", keyboard)

        # Draw a rectangle around the region of interest
        if len(ROI_corners) == 2:
            cv.rectangle(frame, ROI_corners[0], ROI_corners[1], (0, 255, 0), 2)

        cv.imshow("frame", frame)

print("Threshold: ", cv.getTrackbarPos('Threshlod','frame'));
print("Background colour: ", cv.getTrackbarPos('R','frame'), cv.getTrackbarPos('G','frame'), cv.getTrackbarPos('B','frame'));

# Print the corners of the ROI
if len(ROI_corners) == 2:
    print("ROI:", min(ROI_corners[0][0], ROI_corners[1][0]), min(ROI_corners[0][1], ROI_corners[1][1]), max(ROI_corners[0][0], ROI_corners[1][0]), max(ROI_corners[0][1], ROI_corners[1][1]));
