import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

from os import listdir

def get_image_names():
    rows = []
    for y in range(5):
        rows.append([])
        for x in range(5):
            index = y * 5 + x + 1
            if index < 10:
                index = f"0{index}"
            directory = f"Well B1_{index}"
            rows[y].append(listdir(f"./data/{directory}"))
    count = len(rows[0][0])
    images = [[] for _ in range(count)]
    for y in range(5):
        for x in range(5):
            col = rows[y][x]
            if col is None:
                for i in range(count):
                    images[i].append(None)
                continue
            for i in range(count):
                images[i].append(col[i])
    return images

def get_images_from_names(image_names):
    count = len(image_names)
    images = [[] for _ in range(count)]
    for i in range(count):
        for j in range(25):
            index = j + 1
            if index < 10:
                index = f"0{index}"
            images[i].append(cv.imread(f"./data/Well B1_{index}/{image_names[i][j]}"))
    return images

def create_white(max_x, max_y, color = 255):
    return np.zeros((max_y, max_x, 3), np.uint8) + color

def combine_images():
    images = get_images_from_names(get_image_names())
    max_x = images[0][0].shape[1]
    max_y = images[0][0].shape[0]
    final_images = []
    count = len(images)
    for i in range(count):
        white = create_white(max_x * 5, max_y * 5)
        final_images.append(white)
        for y in range(5):
            for x in range(5):
                index = y * 5 + x
                white[y * max_y:(y + 1) * max_y, x * max_x:(x + 1) * max_x] = images[i][index][:]
    return final_images
