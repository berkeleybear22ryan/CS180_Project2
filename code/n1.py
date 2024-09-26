# QUICK TRY FILE FOR PART 1 TO GET FEEL FOR IT
# STANDARDIZED STARTING IN N2.py

import os
import cv2
import numpy as np
from scipy.signal import convolve2d


def load_image(image_path):
    # load image and make sure to convert into grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


def process_image(image, filters, boundary='symm'):
    """
    Step 4: Convolve the image with each filter.
    - No need for manual padding if using `boundary='symm'` in convolve2d.
    """
    results = {}
    for i, F in enumerate(filters, 1):
        # here convolve image with filter F and save result
        convolved_image = convolve2d(image, F, mode='same', boundary=boundary)
        results[f"convolved_F{i}"] = convolved_image
    return results


def save_image(image, file_name, directory):
    os.makedirs(directory, exist_ok=True)
    cv2.imwrite(os.path.join(directory, file_name), image)


def save_thresholded_image(image, threshold, file_name, directory):
    binary_image = (image > threshold).astype(np.uint8) * 255  # need to convert 0/1 to 0/255 for visualization
    save_image(binary_image, file_name, directory)


if __name__ == '__main__':

    output_dir = './render/part1'
    os.makedirs(output_dir, exist_ok=True)

    # define filters (example filters: Dx and Dy for horizontal and vertical edges)
    D_x = np.array([[1, -1]])  # horizontal filter
    D_y = np.array([[1], [-1]])  # vertical filter


    image_number = 1
    image_path = f"./images/{image_number}.jpeg"
    image = load_image(image_path)

    # define filters to use (default: Dx and Dy)
    filters = [D_x, D_y]

    # process the image with each filter
    convolved_images = process_image(image, filters)

    # save the convolved images for each filter
    for filter_name, convolved_image in convolved_images.items():
        save_image(convolved_image, f"{filter_name}.jpeg", output_dir)

    # now combine the results to compute gradient magnitude
    I_x = convolved_images["convolved_F1"]  # result of convolution with Dx
    I_y = convolved_images["convolved_F2"]  # result of convolution with Dy
    gradient_magnitude = np.sqrt(I_x ** 2 + I_y ** 2)

    # now save gradient magnitude image
    save_image(gradient_magnitude, 'gradient_magnitude.jpeg', output_dir)

    # apply thresholding
    threshold_value = 75   # TODO: set threshold value here
    save_thresholded_image(gradient_magnitude, threshold_value, 'binary_edge_map.jpeg', output_dir)

    print(f"Processing completed. Images saved to {output_dir}")
