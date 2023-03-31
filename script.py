from load_images import combine_images
from plants import get_plants, preprocessing
from contours import find_contours
import time
import cv2 as cv
import matplotlib.pyplot as plt
from tracking import intersects, track_contours
import multiprocessing_functions as mpf
from multiprocessing.pool import Pool

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
    #thread_count = 14
    #params = [(images, plants, max_x, max_y, thread_count, i) for i in range(thread_count)]
    #pool = Pool(thread_count)
    #return sum([r for r in pool.map(mpf.load_contours_process, params)])

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

    #start = time.time()
    #contours = find_contours(images[83], plants, max_x, max_y)
    #colored = cv.cvtColor(255 - preprocessing(images[83]).astype('uint8'), cv.COLOR_BGR2RGB)
    #cv.drawContours(colored, contours, -1, (255, 0, 0), 3)
    #plt.imshow(colored)
    #plt.savefig('foo.png')
    #end = time.time()

    #print(end - start)

    #c0 = find_contours(images[0], plants, max_x, max_y)
    #c1 = find_contours(images[1], plants, max_x, max_y)

    #start = time.time()
    #new_c1 = track_contours(c0, c1, max_x, max_y)
    #for i in range(len(c0)):
    #    print(new_c1[i] is None)
    #    print(intersects(c0[i], new_c1[i], max_x, max_y))
    #end = time.time()
    #print(end - start)

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

    contoured_images = [cv.imread(f"./img{'0' if i < 10 else ''}{i}.png") for i in range(len(images))]
    video = cv.VideoWriter("contours.avi", -1, 20.0, (images[0].shape[1], images[0].shape[0]))
    for image in contoured_images:
        video.write(image)
    video.release()