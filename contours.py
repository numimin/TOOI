from plants import preprocessing
import cv2 as cv
from load_images import create_white
import numpy as np

def mask_of_contour(contour, max_x, max_y):
    white = create_white(max_x, max_y)
    cv.fillPoly(white, pts=np.int32([contour.reshape((contour.shape[0], 2))]), color=(255, 0, 0))
    return white

def bounding_box(contour):
    contour = contour.reshape((contour.shape[0], 2))
    first_x = contour[0][0]
    first_y = contour[0][1]
    last_x = 0
    last_y = 0
    for point in contour:
        x = point[0]
        y = point[1]
        if x < first_x:
            first_x = x
        if y < first_y:
            first_y = y
        if x + 1 > last_x:
            last_x = x + 1
        if y + 1 > last_y:
            last_y = y + 1
    return (first_x, first_y, last_x, last_y)

def area_from_contour(contour, max_x, max_y):
    mask = cv.cvtColor(mask_of_contour(contour, max_x, max_y), cv.COLOR_BGR2GRAY)
    
    first_x, first_y, last_x, last_y = bounding_box(contour)
    area = 0
    for y in range(first_y, last_y):
        for x in range(first_x, last_x):
            if mask[y, x] < 255:
                area += 1
    return area

def find_contours(image, plants, max_x, max_y):
    _, objects = cv.threshold(preprocessing(image) - plants, 25, 255, cv.THRESH_BINARY)
    converted = objects.astype('uint8')
    contours, _ = cv.findContours(converted, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    max_lengths = [(0, None)] * 14 #(length, contour)
    for contour in contours:
        contour_size = 1
        for length in contour.shape:
            contour_size = contour_size * length
        max_lengths.append((contour_size, contour))
        max_lengths = list(sorted(max_lengths, key=lambda c: c[0]))[-14:]
    contours = sorted(max_lengths, key = lambda c: area_from_contour(c[1], max_x, max_y))
    return list(map(lambda c: c[1], contours))[-7:]