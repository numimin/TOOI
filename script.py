from load_images import combine_images
from plants import get_plants, preprocessing
from contours import find_contours, area_from_contour, bounding_box
import time
import cv2 as cv
import matplotlib.pyplot as plt
from tracking import intersects, track_contours
import multiprocessing_functions as mpf
from multiprocessing.pool import Pool
import numpy as np

def load_contours(images, plants, max_x, max_y):
    contours = []
    i = 0
    for image in images:
        print(i)
        i += 1
        start = time.time()
        contours.append(find_contours(image, plants, max_x, max_y))
        end = time.time()
        print(end - start)
    print(len(contours))
    return contours

def main():
    start = time.time()
    images = combine_images()
    end = time.time()
    print(end - start)

    start = time.time()
    plants = get_plants(images[0])
    end = time.time()
    print(end - start)

    max_x = images[0].shape[1]
    max_y = images[0].shape[0]

    start = time.time()
    all_contours = load_contours(images, plants, max_x, max_y)
    print(len(all_contours))
    end = time.time()
    print(end - start)

    new_all_contours = [all_contours[0]]
    for i in range(1, len(all_contours)):
        start = time.time()
        print(i)
        new_all_contours.append(track_contours(new_all_contours[i - 1], all_contours[i], max_x, max_y))
        print(len(list(filter(lambda c: not (c is None), new_all_contours[i]))))
        end = time.time()
        print(end - start)

    colors = [(128, 0, 0), 
                (128, 128, 0), 
                (0, 0, 128), 
                (0, 128, 128), 
                (60, 180, 75), 
                (240, 50, 230),
                (70, 240, 240)]
    for i in range(len(images)):
        start = time.time()
        print(i)
        colored = cv.cvtColor(255 - preprocessing(images[i]).astype('uint8'), cv.COLOR_BGR2RGB)
        for j in range(len(colors)):
            if new_all_contours[i][j] is None:
                continue
            cv.drawContours(colored, [new_all_contours[i][j]], -1, colors[j], 3)
        plt.imshow(colored)
        if i < 10:
            i = f"0{i}"
        plt.savefig(f'img{i}.png')
        end = time.time()
        print(end - start)

    areas = [area_from_contour(new_all_contours[i][0], max_x, max_y) for i in range(len(new_all_contours))]  
    plt.plot([i for i in range(len(images))], areas)
    plt.savefig("areas.png")

    plt.clf()
    for j in range(7):
        centers = []
        for i in range(len(new_all_contours)):
            contour = new_all_contours[i][j]
            contour = contour.reshape((contour.shape[0], 2))
            centers.append(sum(contour) / len(contour))
            if i == 0:
                first_center = [centers[0][0], centers[0][1]]
            centers[i] -= first_center
        data = np.array(centers)
        plt.plot(data[:, 0], data[:, 1])

    plt.savefig(f"centers_from_one_point.png")
    plt.draw()
    plt.show()