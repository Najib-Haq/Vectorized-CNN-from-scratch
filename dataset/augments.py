import numpy as np
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt

def rotate(image, angleFrom=-10, angleTo=10):
    '''
    angles in degrees
    '''
    angle = np.random.randint(angleFrom, angleTo)
    return ndimage.rotate(image, angle)


def blur(image):
    sigma = np.random.randint(3, 6)
    return ndimage.gaussian_filter(image, sigma=sigma)


def get_contour_cutout(image, contours, contour_cutout_number):
    image = image.copy()
    no_contours = contours.shape[0]
    # choose random number from 0 to contour_cutout_number
    contour_cutout_number_sample = np.random.randint(2, contour_cutout_number)
    # choose random contours 
    random_contours = np.random.randint(0, no_contours, contour_cutout_number_sample)

    height, width, _ = image.shape
    cutout_height, cutout_width = height*0.02, width*0.02
    for point in contours[random_contours]:
        x, y = point[0][0], point[0][1]
        # cutout the image
        # also check whether the cutout is within the image
        if (x-cutout_width) > 0 and (x+cutout_width) < width and (y-cutout_height) > 0 and (y+cutout_height) < height:
            image[int(y-cutout_height):int(y+cutout_height), int(x-cutout_width):int(x+cutout_width), :] = 0

    return image


def get_number_bb(image, use_bbox,):
    image = image.copy()
    # convert to grayscale and threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    (_, binary) = cv2.threshold(gray, 255//2, 255, cv2.THRESH_OTSU)
    binary = 255-binary

    # apply erosion + dilation to remove noise
    kernel = np.ones((5,5),np.uint8)
    img_opening = cv2.erode(cv2.dilate(binary,kernel,iterations = 1), kernel,iterations = 1)
    # plt.figure()
    # plt.imshow(img_opening, cmap='gray')

    # get bounding box
    contours, _ = cv2.findContours(img_opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))[-1]

    
    if use_bbox:
        bounding_box = cv2.boundingRect(contours)
        return bounding_box, contours
    return _, contours