import cv2
import numpy as np
import matplotlib.pyplot as plt
# import Caliberation, Rectification


def calculate_s_s_dis(pixel_vals_1, pixel_vals_2):
    """Method to calculate the SS distance
    """

    if pixel_vals_1.shape != pixel_vals_2.shape:
        return -1
    return np.sum(abs(pixel_vals_1 - pixel_vals_2))



def get_min_index(y, x, left_local, right, window_size):
    """Method to get the min index intensity values
    """

    local_window = 55
    x_min = max(0, x - local_window)
    x_max = min(right.shape[1], x + local_window)
    min_sad = None
    index_min = None
    first = True
    
    for x in range(x_min, x_max):
        right_local = right[y: y+window_size,x: x+window_size]
        sad = calculate_s_s_dis(left_local, right_local)
        if first:
            min_sad = sad
            index_min = (y, x)
            first = False
        else:
            if sad < min_sad:
                min_sad = sad
                index_min = (y, x)

    return index_min
    
def correspondence(image_left,image_right):
    """Method to obtain the corrsepondance in both the images
    """
    window = 5
    left = np.asarray(image_left)
    right = np.asarray(image_right)
    
    left = left.astype(int)
    right = right.astype(int)
      
    h, w , g = left.shape
    
    disparity = np.zeros((h, w))
    for y in range(window, h-window):
        for x in range(window, w-window):
            left_local = left[y:y + window, x:x + window]
            index_min = get_min_index(y, x, left_local, right, window_size = 5)
            disparity[y, x] = abs(index_min[1] - x)
    
    plt.imshow(disparity, cmap='hot', interpolation='bilinear')
    plt.title('Disparity Hot image')
    plt.savefig('disparity_image_heat.png')
    plt.imshow(disparity, cmap='gray', interpolation='bilinear')
    plt.title('Disparity Plot Gray')
    plt.savefig('disparity_image_gray.png')

    return disparity


def depth_map(disparity, baseline, f):
    """Function to obtain the depth information
    """
    cond1 = np.logical_and(disparity >= 0,disparity < 10)
    cond2 = disparity > 40
    disparity[cond1] = 10
    disparity[cond2] = 40
    depth_map = baseline * f / disparity

    return depth_map
