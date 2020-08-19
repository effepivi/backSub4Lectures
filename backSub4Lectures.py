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


parser = argparse.ArgumentParser(description='This program shows how to use background subtraction.')
# parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
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
green_image = None;
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

while not end_loop:

    addFrame(capture);

    if isinstance(frame, NoneType):
        end_loop = True;
    else:

        if isinstance(green_image, NoneType):
            cv.namedWindow("frame", cv.WINDOW_AUTOSIZE);
            cv.setMouseCallback("frame", clickAndCrop);
            cv.createTrackbar('Threshlod', 'frame', round(255 / 3.5), 255, nothing);
            # cv.createTrackbar('Threshlod R', 'frame', round(255 / 3.5), 255, nothing);
            # cv.createTrackbar('Threshlod G', 'frame', round(255 / 3.5), 255, nothing);
            # cv.createTrackbar('Threshlod B', 'frame', round(255 / 3.5), 255, nothing);

            green_image = np.zeros((frame.shape[0], frame.shape[1], 3), np.float);
            green_image[:] = (0, 1,0);

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
                    cv.imshow('ROI_background', ROI_background);

                    update_ROI = False;

                    green_image = np.zeros((ROI_frame.shape[0], ROI_frame.shape[1], 3), np.float);
                    green_image[:] = (0, 1,0);

                if ROI_background.shape[0] == ROI_frame.shape[0] and ROI_background.shape[1] == ROI_frame.shape[1]:

                    diff = cv.absdiff(cv.pyrDown(cv.pyrDown(ROI_frame)), cv.pyrDown(cv.pyrDown(ROI_background)));

                    grey = cv.normalize(diff, np.zeros(diff.shape), 0, 255, cv.NORM_MINMAX);
                    grey = cv.cvtColor(grey, cv.COLOR_BGR2GRAY);

                    # grey_r= cv.normalize(diff_r, np.zeros(diff_r.shape), 0, 255, cv.NORM_MINMAX).astype('uint8');
                    # # grey_r = cv.cvtColor(grey_r, cv.COLOR_BGR2GRAY);
                    #
                    # grey_g= cv.normalize(diff_g, np.zeros(diff_g.shape), 0, 255, cv.NORM_MINMAX).astype('uint8');
                    # # grey_g = cv.cvtColor(grey_g, cv.COLOR_BGR2GRAY);
                    #
                    # grey_b= cv.normalize(diff_b, np.zeros(diff_b.shape), 0, 255, cv.NORM_MINMAX).astype('uint8');
                    # # grey_b = cv.cvtColor(grey_b, cv.COLOR_BGR2GRAY);

                    #
                    cv.imshow('diff', grey);
                    # cv.imshow('diff_g', grey_g);
                    # cv.imshow('diff_b', grey_b);
                    #
                    # cv.imshow('ROI_frame_r', ROI_frame_r);
                    # cv.imshow('ROI_frame_g', ROI_frame_g);
                    # cv.imshow('ROI_frame_b', ROI_frame_b);

                    ret, foreground_mask = cv.threshold(grey, cv.getTrackbarPos('Threshlod','frame'), 255, cv.THRESH_BINARY);
                    # ret, foreground_mask_r = cv.threshold(grey_r, cv.getTrackbarPos('Threshlod R','frame'), 255, cv.THRESH_BINARY);
                    # ret, foreground_mask_g = cv.threshold(grey_g, cv.getTrackbarPos('Threshlod G','frame'), 255, cv.THRESH_BINARY);
                    # ret, foreground_mask_b = cv.threshold(grey_b, cv.getTrackbarPos('Threshlod B','frame'), 255, cv.THRESH_BINARY);
                    cv.imshow('grey', grey);
                    cv.imshow('foreground_mask', foreground_mask);
                    # cv.imshow('foreground_mask_r', foreground_mask_r);
                    # cv.imshow('foreground_mask_g', foreground_mask_g);
                    # cv.imshow('foreground_mask_b', foreground_mask_b);
                    # cv.imshow('foreground_mask', foreground_mask);

                    foreground_mask = cv.medianBlur(foreground_mask, 1);
                    cv.imshow('medianBlur', foreground_mask);
                    foreground_mask = cv.dilate(foreground_mask, kernel2, iterations = 3);
                    cv.imshow('dilate', foreground_mask);
                    foreground_mask = cv.morphologyEx(foreground_mask, cv.MORPH_CLOSE, kernel1);
                    cv.imshow('MORPH_CLOSE', foreground_mask);
                    foreground_mask = cv.morphologyEx(foreground_mask, cv.MORPH_OPEN, kernel2);
                    cv.imshow('MORPH_OPEN', foreground_mask);
                    foreground_mask = foreground_mask.astype('float') / 255;


                    # foreground_mask_r = cv.dilate(foreground_mask_r, kernel2, iterations = 3);
                    # foreground_mask_r = cv.morphologyEx(foreground_mask_r, cv.MORPH_CLOSE, kernel1);
                    # foreground_mask_r = cv.morphologyEx(foreground_mask_r, cv.MORPH_OPEN, kernel2);
                    #
                    # foreground_mask_g = cv.dilate(foreground_mask_g, kernel2, iterations = 3);
                    # foreground_mask_g = cv.morphologyEx(foreground_mask_g, cv.MORPH_CLOSE, kernel1);
                    # foreground_mask_g = cv.morphologyEx(foreground_mask_g, cv.MORPH_OPEN, kernel2);
                    #
                    # foreground_mask_b = cv.dilate(foreground_mask_b, kernel2, iterations = 3);
                    # foreground_mask_b = cv.morphologyEx(foreground_mask_b, cv.MORPH_CLOSE, kernel1);
                    # foreground_mask_b = cv.morphologyEx(foreground_mask_b, cv.MORPH_OPEN, kernel2);
                    #
                    # foreground_mask_r = cv.medianBlur(foreground_mask_r, 3);
                    # foreground_mask_g = cv.medianBlur(foreground_mask_g, 3);
                    # foreground_mask_b = cv.medianBlur(foreground_mask_b, 3);

                    alpha3 = cv.pyrUp(cv.pyrUp(np.stack([foreground_mask]*3, axis=2)));
                    alpha3 = alpha3[0:ROI_frame.shape[0], 0:ROI_frame.shape[1]];


                    #
                    # cv.imshow('foreground_mask', alpha3);
                    # print(foreground_mask_g.shape, ROI_frame.shape)
                    cv.imshow('ROI_frame', ROI_frame);
                    cv.imshow('chroma key', alpha3 * ROI_frame / (255) + (1-alpha3) * green_image);

        keyboard = cv.waitKey(30)
        if keyboard == 'b' or keyboard == ord('b'):
            for i in range(tot_frame):
                ret, temp = capture.read();
                temp = temp.astype("float")
                if i == 0:
                    background = temp;
                else:
                    background = np.add(background, temp);

            background = np.divide(background, tot_frame);
            background = background.astype('uint8');

            ROI_background = cv.GaussianBlur(background, (5,5),0);
            cv.imshow('background', background);

            # background_r, background_g, background_b = normaliseRGB(frame);
            #
            # ROI_background_r = cv.GaussianBlur(background_r, (5,5),0);
            # ROI_background_g = cv.GaussianBlur(background_g, (5,5),0);
            # ROI_background_b = cv.GaussianBlur(background_b, (5,5),0);

        elif keyboard == 'q' or keyboard == ord('q') or keyboard == 27:
            end_loop = True;

        # draw a rectangle around the region of interest
        if len(ROI_corners) == 2:
            cv.rectangle(frame, ROI_corners[0], ROI_corners[1], (0, 255, 0), 2)

        cv.imshow("frame", frame)


#
# # if there are two reference points, then crop the region of interest
# # from teh image and display it
# if len(refPt) == 2:
# 	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
# 	cv2.imshow("ROI", roi)
# 	cv2.waitKey(0)
