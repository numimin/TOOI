from load_images import combine_images
from plants import get_plants, preprocessing
from contours import find_contours
import time
import cv2 as cv
import matplotlib.pyplot as plt
from tracking import intersects, track_contours

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
    contours = find_contours(images[83], plants, max_x, max_y)
    colored = cv.cvtColor(255 - preprocessing(images[83]).astype('uint8'), cv.COLOR_BGR2RGB)
    cv.drawContours(colored, contours, -1, (255, 0, 0), 3)
    plt.imshow(colored)
    plt.savefig('foo.png')
    end = time.time()
    print(end - start)

    c0 = find_contours(images[0], plants, max_x, max_y)
    c1 = find_contours(images[1], plants, max_x, max_y)

    start = time.time()
    new_c1 = track_contours(c0, c1, max_x, max_y)
    for i in range(len(c0)):
        print(new_c1[i] is None)
        print(intersects(c0[i], new_c1[i], max_x, max_y))
    end = time.time()
    print(end - start)
