from load_images import combine_images
from plants import get_plants
import time

def main():
    start = time.time()
    images = combine_images()
    end = time.time()
    print(end - start)

    start = time.time()
    plants = get_plants(images[0])
    end = time.time()
    print(end - start)
    