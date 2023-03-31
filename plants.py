from multiprocessing.pool import Pool
import multiprocessing_functions as mpf
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from load_images import create_white

def preprocessing(image):
    return 255 - cv.cvtColor(image,cv.COLOR_BGR2GRAY)

def morph_open(img, morph_size):
    element = cv.getStructuringElement(cv.MORPH_RECT, (2 * morph_size + 1, 2 * morph_size + 1), (morph_size, morph_size))
    return cv.morphologyEx(img, cv.MORPH_OPEN, element)

def retain_neighbored(image, threshold):
    thread_count = 14
    params = [(image, threshold, 1, 1, image.shape[1] - 1, image.shape[0] - 1, thread_count, i) for i in range(thread_count)]
    pool = Pool(thread_count)
    return sum([r for r in pool.imap_unordered(mpf.retain_neighbored_process, params)])

def cut(image, x, y, w, h):
    white = create_white(image.shape[1], image.shape[0], 0)
    cv.rectangle(white, (x, y), (x + w, y + h), (1, 1, 1), -1)
    white = cv.cvtColor(white, cv.COLOR_BGR2GRAY)
    return white * image

def delete_background(cut_image, neighbors_coefficient, open_coefficient):
    deleted = retain_neighbored(cut_image, neighbors_coefficient)
    return morph_open(deleted, open_coefficient)

def get_plants(image):
    preprocessed = preprocessing(image)
    trichoplaxes = []
    trichoplaxes.append(0.4 * delete_background(cut(preprocessed, 1400, 2900, 500, 700), 100, 5))
    trichoplaxes.append(0.4 * delete_background(cut(preprocessed, 4520, 380, 680, 880), 40, 3))
    trichoplaxes.append(0.5 * delete_background(cut(preprocessed, 3500, 1850, 780, 450), 70, 4))
    trichoplaxes.append(0.4 * delete_background(cut(preprocessed, 1500, 4500, 450, 400), 100, 3))
    trichoplaxes.append(0.4 * delete_background(cut(preprocessed, 4650, 2000, 500, 600), 20, 3))
    trichoplaxes.append(0.7 * delete_background(cut(preprocessed, 4850, 3100, 400, 700), 1, 1))
    trichoplaxes.append(0.2 * delete_background(cut(preprocessed, 6400, 3550, 600, 500), 135, 5))
    return preprocessed - sum(trichoplaxes)