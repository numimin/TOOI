from contours import mask_of_contour, bounding_box
import cv2 as cv
import multiprocessing_functions as mpf
from multiprocessing.pool import Pool

def intersects(lhs, rhs, max_x, max_y):
    lmask = cv.cvtColor(mask_of_contour(lhs, max_x, max_y), cv.COLOR_BGR2GRAY)
    lfirst_x, lfirst_y, llast_x, llast_y = bounding_box(lhs)
    rmask = cv.cvtColor(mask_of_contour(rhs, max_x, max_y), cv.COLOR_BGR2GRAY)
    rfirst_x, rfirst_y, rlast_x, rlast_y = bounding_box(rhs)

    if lfirst_x > rlast_x:
        return False
    if rfirst_x > llast_x:
        return False
    if lfirst_y > rlast_y:
        return False
    if rfirst_y > llast_y:
        return False

    first_x = min(lfirst_x, rfirst_x)
    first_y = min(lfirst_y, rfirst_y)
    last_x = max(llast_x, rlast_x)
    last_y = max(llast_y, rlast_y)

    thread_count = 14
    params = [(lmask, rmask, first_x, first_y, last_x, last_y, thread_count, i) for i in range(thread_count)]
    pool = Pool(thread_count)
    return any([r for r in pool.imap_unordered(mpf.retain_neighbored_process, params)])