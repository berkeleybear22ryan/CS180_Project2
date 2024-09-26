import cv2
import numpy as np
import os
from utility import open_image, save_image, normalize, convolve_v3, gaussian_filter

if __name__ == '__main__':

    # Constants for the project
    # image1_path = './images/33.jpeg'
    # image2_path = './images/32.jpeg'
    # output_path = './render/part5/'

    image1_path = './images/38.jpeg'
    image2_path = './images/36.jpeg'
    output_path = './render/part5_1/'



    # make sure to create output directories if they do not exist
    os.makedirs(output_path + 'gaussian_stack/', exist_ok=True)
    os.makedirs(output_path + 'laplacian_stack/', exist_ok=True)

    # load images ... confirm normalization -- seems fine
    apple = normalize(open_image(image1_path))
    orange = normalize(open_image(image2_path))

    # now ... generate gaussian and laplacian stacks
    def generate_gaussian_stack(image, levels, kernel_size=5, sigma=1.0):
        gaussian_stack = [image]
        G = gaussian_filter(kernel_size, sigma)
        current_image = image
        for _ in range(levels - 1):
            current_image = convolve_v3(current_image, G)
            gaussian_stack.append(current_image)
        return gaussian_stack

    def generate_laplacian_stack(gaussian_stack):
        laplacian_stack = []
        for i in range(len(gaussian_stack) - 1):
            next_gaussian = gaussian_stack[i + 1]
            current_gaussian = gaussian_stack[i]
            # now ... do direct subtraction without upsampling, ensuring same size -- as stated
            laplacian = current_gaussian - next_gaussian
            laplacian_stack.append(laplacian)
        # note ... the last level of the Laplacian stack is the last level of the Gaussian stack
        laplacian_stack.append(gaussian_stack[-1])
        return laplacian_stack

    # create Gaussian stacks
    # TODO: change me
    levels = 10
    apple_gaussian_stack = generate_gaussian_stack(apple, levels)
    orange_gaussian_stack = generate_gaussian_stack(orange, levels)

    # save Gaussian stack levels for visualization, note names are swapped but wanted to match orientation of sample
    for i, (gaussian_apple, gaussian_orange) in enumerate(zip(apple_gaussian_stack, orange_gaussian_stack)):
        save_image(output_path + f'gaussian_stack/apple_gaussian_level_{i}.png', normalize(gaussian_apple))
        save_image(output_path + f'gaussian_stack/orange_gaussian_level_{i}.png', normalize(gaussian_orange))

    # now ... create Laplacian stacks
    apple_laplacian_stack = generate_laplacian_stack(apple_gaussian_stack)
    orange_laplacian_stack = generate_laplacian_stack(orange_gaussian_stack)

    # save Laplacian stack levels for visualization
    for i, (laplacian_apple, laplacian_orange) in enumerate(zip(apple_laplacian_stack, orange_laplacian_stack)):
        save_image(output_path + f'laplacian_stack/apple_laplacian_level_{i}.png', normalize(laplacian_apple))
        save_image(output_path + f'laplacian_stack/orange_laplacian_level_{i}.png', normalize(laplacian_orange))

    # generate a smoother mask
    def generate_smooth_mask(shape, transition_width):
        mask = np.zeros(shape, dtype=np.float32)
        center = shape[1] // 2
        mask[:, center - transition_width//2:center + transition_width//2] = np.linspace(0, 1, transition_width)
        mask[:, center + transition_width//2:] = 1
        return mask

    # generate a smooth gradient mask
    # TODO: change me
    transition_width = 100  # Adjust this value to control the smoothness of the transition
    smooth_mask = generate_smooth_mask(apple.shape[:2], transition_width)

    # now important ... apply Gaussian blur to the mask to smooth it further
    # TODO: change me
    # must be odd number so that kernel has center -- check
    k = 21
    sig = 10
    smooth_mask = cv2.GaussianBlur(smooth_mask, (k, k), sig)

    # here ... blend Laplacian stacks using the smooth mask
    def blend_laplacian_stacks(stack1, stack2, mask):
        blended_stack = []
        for l1, l2 in zip(stack1, stack2):
            blended = l1 * mask[:, :, np.newaxis] + l2 * (1 - mask[:, :, np.newaxis])
            blended_stack.append(blended)
        return blended_stack

    # now create blended Laplacian stack
    blended_laplacian_stack = blend_laplacian_stacks(apple_laplacian_stack, orange_laplacian_stack, smooth_mask)

    # and now reconstruct image from Laplacian stack
    def reconstruct_from_laplacian_stack(laplacian_stack):
        # direct addition, no upsampling
        current_image = laplacian_stack[-1]
        for i in range(len(laplacian_stack) - 2, -1, -1):
            current_image = laplacian_stack[i] + current_image
        return current_image

    # reconstruct the final blended image
    blended_image = reconstruct_from_laplacian_stack(blended_laplacian_stack)

    # save blended image ... check this
    save_image(output_path + 'blended_image.png', blended_image)

    # save individual levels of blended Laplacian stack for visualization ... just in case
    for i, lap in enumerate(blended_laplacian_stack):
        save_image(output_path + f'blended_laplacian_level_{i}.png', normalize(lap))

    print("Blending complete. Check the output folder for results.")
