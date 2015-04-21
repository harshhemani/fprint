"""
@author: Harsh Hemani
@date:   21/04/2015
@SoP:    Mainly here to provide `get_orientation_map` function
         This function takes an image matrix as input (gray level)
         and returns an orientation-map, i.e., a matrix that contains
         the orientation (theta) value for each pixel in the image.
         Orientation denotes the tangent of the ridges at the point.
"""
import sys
import Image
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
from pylab import imshow, show

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

def get_orientation_map(image_matrix):
    """
    Input an image matrix (grayscale)
    and
    Outputs the direction of ridge tanget at each pixel
    """
    bin_img_mx = binarize(image_matrix)
    Gx = ndimage.sobel(bin_img_mx, 0)
    Gy = ndimage.sobel(bin_img_mx, 1)
    # grad_img = np.hypot(Gx, Gy)
    # grad_img *= 255.0 / np.max(grad_img)
    # imshow(grad_img)
    # show()
    Gxx = np.zeros(Gx.shape)
    Gyy = np.zeros(Gy.shape)
    theta = np.zeros(Gx.shape)
    W = 1
    for i in range(Gxx.shape[0]):
        for j in range(Gxx.shape[1]):
            lower_k = i - W
            lower_l = j - W
            upper_k = i + W
            upper_l = j + W
            if lower_k < 0:
                lower_k = 0
            if lower_l < 0:
                lower_l = 0
            if upper_k >= Gxx.shape[0]:
                upper_k = Gxx.shape[0] - 1
            if upper_l >= Gxx.shape[1]:
                upper_l = Gxx.shape[1] - 1
            for k in range(lower_k, upper_k+1):
                for l in range(lower_l, upper_l+1):
                    Gxx[i][j] += (Gx[k][l] ** 2) - (Gy[k][l] ** 2)
                    Gyy[i][j] += 2.0 * Gx[k][l] * Gy[k][l]
            if abs(Gxx[i][j]) <= 1.0E-10:
                theta[i][j] = np.pi / 2.0
            else:
                theta[i][j] = 0.5 * np.arctan(Gyy[i][j]/Gxx[i][j])
    # now average the thetas, cuz image is noisy (eg: broken ridges)
    gauss_line = np.array([1, 4, 6, 4, 1])[:, np.newaxis]
    kernel = (1.0 / 256) * np.dot(gauss_line, gauss_line.T)
    conv_numerator = convolve2d(np.sin(2*theta), kernel, mode='same')
    conv_denomenat = convolve2d(np.cos(2*theta), kernel, mode='same')
    theta_prime = 0.5 * np.arctan(conv_numerator / conv_denomenat)
    return theta_prime

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print '\nSyntax:\n\tpython', sys.argv[0], '<path/to/image>\n'
        sys.exit()
    image_path = sys.argv[1]
    print 'Loading image..'
    image_mx = np.asarray(Image.open(image_path).convert('L'))
    print 'Image loaded.'
    theta_map = get_orientation_map(image_mx)
    theta_map_min = np.min(theta_map)
    theta_map_max = np.max(theta_map)
    theta_im = 255 * (theta_map - theta_map_min*np.ones(theta_map.shape)) / theta_map_max
    imshow(theta_im, 'gray')
    show()
    print 'Done!'

