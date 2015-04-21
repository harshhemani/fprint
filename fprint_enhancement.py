import sys
import Image
import numpy as np
from scipy.signal import convolve2d
from pylab import imshow, figure, show

def invert_binary_vals(img):
    max_val = np.max(img)
    zero_indices = img==0
    new_img = np.zeros(img.shape)
    new_img[zero_indices] = max_val
    return new_img

def binarize(img, threshold=150):
    """
    Convert image into a binary image
    With white denoted by 0
    and
    Black denoted by 255
    """
    new_img = np.ones(img.shape) * 255
    white_indices = img > threshold
    new_img[white_indices] = 0
    return new_img

def enhance_fprint_1(matrix, block_size=10, alpha=0.7):
    """
    Enhance image using locally calculating average
    and then thresholding against it.
    Block size determines the window for computing average
    Thresholding is done using computed mean times alpha
    assuming ridges are represented by high values
    """
    for i in range((matrix.shape[0]/block_size)+1):
        for j in range((matrix.shape[1]/block_size)+1):
            start_i = i * block_size
            end_i = (i+1) * block_size
            start_j = j * block_size
            end_j = (j+1) * block_size
            roi = matrix[start_i:end_i, start_j:end_j]
            threshold = np.mean(roi) * alpha
            acceptable_indices = roi >= threshold
            new_roi = np.zeros(roi.shape)
            new_roi[acceptable_indices] = 255
            matrix[start_i:end_i, start_j:end_j] = new_roi
    return matrix

def enhance_fprint(matrix, method):
    matrix = binarize(matrix)
    if method=='local-avg-thresh':
        return enhance_fprint_1(matrix)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print '\nSyntax:\n\tpython', sys.argv[0], '<path/to/image>\n'
        sys.exit()
    image_path = sys.argv[1]
    image_matrix = np.asarray(Image.open(image_path).convert('L'))
    enhanced_image = enhance_fprint(image_matrix, method='local-avg-thresh')
    fig = figure()
    fig.add_subplot(1, 2, 1)
    imshow(image_matrix, 'gray')
    fig.add_subplot(1, 2, 2)
    imshow(invert_binary_vals(enhanced_image), 'gray')
    show()
