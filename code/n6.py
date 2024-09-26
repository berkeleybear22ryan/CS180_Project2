from utility import *
import cv2
import numpy as np
import os

if __name__ == '__main__':

    image1_path = './images/47.jpeg'
    image2_path = './images/46.jpeg'
    custom_mask_path = './masks/m9.png'
    output_path = './render/part6_8/'

    # make sure to create output directories if they do not exist
    os.makedirs(output_path, exist_ok=True)

    # load images
    img1 = normalize(open_image(image1_path))
    img2 = normalize(open_image(image2_path))

    # upddate ... options dictionary for different configurations
    options = {
        'gaussian_default': {'filter_type': 'gaussian', 'k': 5, 'sigma': 1.0},
        'gaussian_smooth': {'filter_type': 'gaussian', 'k': 50, 'sigma': 3.0},
        'sobel_x': {'filter_type': 'sobel_x'},
        'sobel_y': {'filter_type': 'sobel_y'},
        'laplacian': {'filter_type': 'laplacian'}
    }

    # TODO: select configuration option ...
    selected_option = 'gaussian_smooth'  # IMPORTANT ... change this to one of the keys in the options dictionary

    # applying the selected option
    if selected_option in options:
        config = options[selected_option]
    else:
        raise ValueError("Unsupported option selected. Choose from: {}".format(list(options.keys())))

    # define filter parameters based on selected configuration
    filter_type = config['filter_type']
    k = config.get('k', 5)
    sigma = config.get('sigma', 1.0)


    # function to load a custom mask
    def load_custom_mask(mask_path, image_shape):
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Custom mask file not found: {mask_path}")

        custom_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if custom_mask is None:
            raise ValueError(f"Failed to load custom mask from: {mask_path}")

        # need to ... resize mask to match the image shape
        custom_mask_resized = cv2.resize(custom_mask, (image_shape[1], image_shape[0]))

        # normalize the mask to the range [0, 1]
        return custom_mask_resized / 255.0


    # generate various mask options
    def generate_vertical_gradient_mask(image_shape, blend_width):
        mask = np.zeros(image_shape[:2], dtype=np.float32)
        center = image_shape[1] // 2
        blend_start = center - blend_width // 2
        blend_end = center + blend_width // 2
        mask[:, :blend_start] = 0
        mask[:, blend_start:blend_end] = np.linspace(0, 1, blend_end - blend_start)
        mask[:, blend_end:] = 1
        return mask


    def generate_horizontal_gradient_mask(image_shape, blend_width):
        mask = np.zeros(image_shape[:2], dtype=np.float32)
        center = image_shape[0] // 2
        blend_start = center - blend_width // 2
        blend_end = center + blend_width // 2
        mask[:blend_start, :] = 0
        mask[blend_start:blend_end, :] = np.linspace(0, 1, blend_end - blend_start)[:, None]
        mask[blend_end:, :] = 1
        return mask


    def generate_elliptical_mask(image_shape, axes_ratio=0.5):
        mask = np.zeros(image_shape[:2], dtype=np.float32)
        center = (image_shape[1] // 2, image_shape[0] // 2)
        axes = (int(image_shape[1] * axes_ratio), int(image_shape[0] * axes_ratio))
        mask = cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
        return mask


    def generate_circular_mask(image_shape, radius_ratio=0.3):
        mask = np.zeros(image_shape[:2], dtype=np.float32)
        center = (image_shape[1] // 2, image_shape[0] // 2)
        radius = int(min(image_shape[:2]) * radius_ratio)
        mask = cv2.circle(mask, center, radius, 1, -1)
        return mask


    def smooth_mask(mask, k, sigma=5.0):
        # make sure okay to remove + odd number k check ... ensure mask is in float format and normalized to [0, 1] if it's not already
        # if mask.max() > 1:
        #     mask = mask / 255.0

        # apply gaussian blur
        smoothed_mask = cv2.GaussianBlur(mask, (k, k), sigma)

        return smoothed_mask


    def generate_sobel_edge_mask(image):
        # convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel_edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        _, edge_mask = cv2.threshold(sobel_edges, 50, 1, cv2.THRESH_BINARY)
        return edge_mask


    def generate_refined_tattoo_mask(image, canny_threshold1=50, canny_threshold2=150, clahe_clip=2.0, clahe_tile_grid_size=8):
        # ... convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # apply (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size))
        enhanced = clahe.apply(gray)

        # apply Gaussian blur to reduce noise and help with edge detection
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 1)

        # now ... canny edge detection with refined thresholds
        edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)

        # now ... find contours and fill them to create a filled mask
        filled_mask = np.zeros_like(edges, dtype=np.uint8)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(filled_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # IMPORTANT: invert the filled mask to get the desired result ... so that proper img gets mapped
        inverted_mask = cv2.bitwise_not(filled_mask)

        # TODO: optional: Apply a slight blur to smooth the final mask
        smoothed_inverted_mask = cv2.GaussianBlur(inverted_mask, (5, 5), 1)

        return smoothed_inverted_mask / 255.0  # CHECK: normalize to 0-1 range

    def generate_enhanced_outline_mask(image, dilation_iter=2, min_contour_area=1):
        # ... convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # canny edge detection
        # print(gray)
        edges = cv2.Canny(image, 50, 500)

        # dilate edges to make them thicker
        dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=dilation_iter)

        # apply morphological closing to close gaps
        closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # create an empty mask for filling contours
        mask = np.zeros_like(closed_edges, dtype=np.uint8)

        # find contours and fill them
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # need to filter out small contours based on area
            if cv2.contourArea(contour) > min_contour_area:
                # now check solidity to avoid filling small gaps and noise
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = cv2.contourArea(contour) / hull_area if hull_area > 0 else 0

                if solidity > 0.8:  # only fill if contour has high solidity ... confirm
                    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # further dilate and smooth the mask
        final_mask = cv2.dilate(mask, np.ones((1, 1), np.uint8), iterations=1)
        final_mask = cv2.GaussianBlur(final_mask, (3, 3), 1)

        return final_mask / 255.0  # normalize to 0-1 range


    # choose a mask option
    # got this best one from generate_refined_tattoo_mask
    mask_type = 'custom'  # ... change to 'custom', 'vertical_gradient', 'horizontal_gradient', 'elliptical', 'circular', 'sobel_edge', 'outline'

    # generate mask based on the chosen type
    if mask_type == 'vertical_gradient':
        blend_width = 100
        mask = generate_vertical_gradient_mask(img1.shape, blend_width)
    elif mask_type == 'horizontal_gradient':
        blend_width = 100
        mask = generate_horizontal_gradient_mask(img1.shape, blend_width)
    elif mask_type == 'elliptical':
        mask = generate_elliptical_mask(img1.shape)
    elif mask_type == 'circular':
        mask = generate_circular_mask(img1.shape)
    elif mask_type == 'sobel_edge':
        mask = generate_sobel_edge_mask(img1)
    elif mask_type == 'outline':
        mask = generate_enhanced_outline_mask(img1)
    elif mask_type == 'custom':
        mask = load_custom_mask(custom_mask_path, img1.shape)
    elif mask_type == "tattoo":
        mask = generate_refined_tattoo_mask(img2)
    else:
        raise ValueError(
            "Unsupported mask type. Choose from ['custom', 'vertical_gradient', 'horizontal_gradient', 'elliptical', 'circular', 'sobel_edge', 'outline']")

    # normalize mask to be in the range [0, 255]
    mask = normalize(mask * 255)

    # TODO: can comment out for other images -- trying for the tattoo one
    # mask = smooth_mask(mask, 41, 1001)
    # mask = normalize(mask * 255)

    # now need to save the mask for reference
    save_image(output_path + 'mask.png', mask)


    # function to downsample manually
    def downsample(image):
        return image[::2, ::2]  # take every second pixel


    # function to upsample manually with proper channel handling
    def upsample(image):
        if len(image.shape) == 3:  # need to ... check if the image has multiple channels (e.g., RGB)
            upsampled_image = np.zeros((image.shape[0] * 2, image.shape[1] * 2, image.shape[2]), dtype=image.dtype)
        else:
            upsampled_image = np.zeros((image.shape[0] * 2, image.shape[1] * 2), dtype=image.dtype)

        upsampled_image[::2, ::2] = image  # will insert original pixels into new image

        # now apply Gaussian filter to smooth the upsampled image
        gaussian_kernel = gaussian_filter(5, 1.0)
        return convolve_v3(upsampled_image, gaussian_kernel)


    # function to select a filter based on the configuration
    def get_filter(filter_type, k=5, sigma=1.0):
        if filter_type == 'gaussian':
            return gaussian_filter(k, sigma)
        elif filter_type == 'sobel_x':
            return np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ])
        elif filter_type == 'sobel_y':
            return np.array([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ])
        elif filter_type == 'laplacian':
            return np.array([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ])
        else:
            raise ValueError("Unsupported filter type. Choose from ['gaussian', 'sobel_x', 'sobel_y', 'laplacian']")


    # generate Gaussian pyramid manually with downsampling
    def generate_gaussian_pyramid(image, levels, filter_type='gaussian', k=5, sigma=1.0):
        gaussian_pyramid = [image]
        current_image = image
        selected_filter = get_filter(filter_type, k, sigma)  # get the selected filter
        for _ in range(levels - 1):
            current_image = convolve_v3(current_image, selected_filter)
            current_image = downsample(current_image)
            gaussian_pyramid.append(current_image)
        return gaussian_pyramid


    # generate Laplacian pyramid manually
    def generate_laplacian_pyramid(gaussian_pyramid):
        laplacian_pyramid = []
        for i in range(len(gaussian_pyramid) - 1):
            current_gaussian = gaussian_pyramid[i]
            next_gaussian = gaussian_pyramid[i + 1]
            next_gaussian_upsampled = upsample(next_gaussian)
            # need to make sure both images have the same size before subtracting
            if current_gaussian.shape != next_gaussian_upsampled.shape:
                next_gaussian_upsampled = cv2.resize(next_gaussian_upsampled,
                                                     (current_gaussian.shape[1], current_gaussian.shape[0]))
            laplacian = current_gaussian - next_gaussian_upsampled
            laplacian_pyramid.append(laplacian)
        laplacian_pyramid.append(gaussian_pyramid[-1])  # CHECK: the last level is the smallest Gaussian level
        return laplacian_pyramid


    # set number of levels/depth
    levels = 20  # increase the number of levels for more detailed blending

    # NOTE: create Gaussian and Laplacian pyramids for the images and the mask
    img1_gaussian_pyramid = generate_gaussian_pyramid(img1, levels, filter_type=filter_type, k=k, sigma=sigma)
    img2_gaussian_pyramid = generate_gaussian_pyramid(img2, levels, filter_type=filter_type, k=k, sigma=sigma)
    mask_gaussian_pyramid = generate_gaussian_pyramid(mask, levels, filter_type=filter_type, k=k, sigma=sigma)

    img1_laplacian_pyramid = generate_laplacian_pyramid(img1_gaussian_pyramid)
    img2_laplacian_pyramid = generate_laplacian_pyramid(img2_gaussian_pyramid)

    # will always ... save Gaussian and Laplacian stacks for visualization
    for i in range(levels):
        save_image(output_path + f'img1_gaussian_level_{i}.png', normalize(img1_gaussian_pyramid[i]))
        save_image(output_path + f'img2_gaussian_level_{i}.png', normalize(img2_gaussian_pyramid[i]))
        save_image(output_path + f'mask_gaussian_level_{i}.png', normalize(mask_gaussian_pyramid[i]))
        save_image(output_path + f'img1_laplacian_level_{i}.png', normalize(img1_laplacian_pyramid[i]))
        save_image(output_path + f'img2_laplacian_level_{i}.png', normalize(img2_laplacian_pyramid[i]))


    # blend the Laplacian pyramids using the Gaussian mask pyramid
    def blend_laplacian_pyramids_with_mask(stack1, stack2, mask_pyramid):
        blended_pyramid = []
        for l1, l2, mask in zip(stack1, stack2, mask_pyramid):
            # expand the mask to 3 channels if the images are RGB
            if len(l1.shape) == 3 and mask.ndim == 2:
                mask = np.stack([mask] * 3, axis=-1)  # convert single channel mask to 3 channels

            # blend using the expanded mask
            blended = l1 * (mask / 255.0) + l2 * (1 - (mask / 255.0))
            blended_pyramid.append(blended)
        return blended_pyramid


    # blend the Laplacian pyramids with the mask Gaussian pyramid
    blended_laplacian_pyramid = blend_laplacian_pyramids_with_mask(
        img1_laplacian_pyramid, img2_laplacian_pyramid, mask_gaussian_pyramid
    )


    # reconstruct the blended image from the blended Laplacian pyramid
    def reconstruct_from_laplacian_pyramid(laplacian_pyramid):
        current_image = laplacian_pyramid[-1]
        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            current_image = upsample(current_image)
            # always need to make sure both images have the same size before adding
            if laplacian_pyramid[i].shape != current_image.shape:
                current_image = cv2.resize(current_image,
                                           (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0]))
            current_image = laplacian_pyramid[i] + current_image
        return current_image


    # reconstruct the final blended image
    blended_image = reconstruct_from_laplacian_pyramid(blended_laplacian_pyramid)

    # save the final blended image
    save_image(output_path + 'blended_image.png', normalize(blended_image))

    # save blended Laplacian pyramid levels for visualization
    for i, lap in enumerate(blended_laplacian_pyramid):
        save_image(output_path + f'blended_laplacian_level_{i}.png', normalize(lap))

    print("Multiresolution blending complete. Check the output folder for results.")
