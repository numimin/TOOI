import numpy as np

def retain_neighbored_process(params):
    image, threshold, first_x, first_y, last_x, last_y, thread_count, thread_number = params
    out = np.zeros(image.shape, np.uint8)
    y_count = (last_y - first_y) // thread_count
    actual_first_y = first_y + y_count * thread_number
    actual_last_y = first_y + y_count * (thread_number + 1)
    if thread_number == thread_count - 1:
        actual_last_y = last_y
    for y in range(actual_first_y, actual_last_y):
        for x in range(first_x, last_x):
            hasAllNeighbors = True
            hasAllNeighbors = hasAllNeighbors and image[y + 1, x + 1] > threshold
            hasAllNeighbors = hasAllNeighbors and image[y + 1, x - 1] > threshold            
            hasAllNeighbors = hasAllNeighbors and image[y - 1, x - 1] > threshold            
            hasAllNeighbors = hasAllNeighbors and image[y - 1, x + 1] > threshold   
            if hasAllNeighbors:
                out[y, x] = image[y, x]
    return out

def intersects_process(params):
    lmask, rmask, first_x, first_y, last_x, last_y, thread_count, thread_number = params
    y_count = (last_y - first_y) // thread_count
    actual_first_y = first_y + y_count * thread_number
    actual_last_y = first_y + y_count * (thread_number + 1)
    if thread_number == thread_count - 1:
        actual_last_y = last_y
    for y in range(actual_first_y, actual_last_y):
        for x in range(first_x, last_x):
            if lmask[y, x] < 255 and rmask[y, x] < 255:
                return True
    return False