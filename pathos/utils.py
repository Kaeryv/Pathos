#!/bin/python3

import numpy as np


import dlib
import cv2
from imutils import face_utils



def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)


def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return x - x_off, x + width + x_off, y - y_off, y + height + y_off


def draw_text(coordinates, image_array, text: str, color, x_offset=0, y_offset=0,
              font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def format_image(bgr_image):
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    gray_image = cv2.equalizeHist(gray_image)
    try:
        gray_image = cv2.resize(gray_image, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[Error] While resizing")
        return None

    return gray_image


