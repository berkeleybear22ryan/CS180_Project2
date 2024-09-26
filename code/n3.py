from utility import *


def unsharp_mask(image, gaussian_kernel, alpha=1.5):
    # blur the original image
    blurred = convolve_v3(image, gaussian_kernel, mode='same', boundary='symm')
    # get high-frequency components
    high_freq = image - blurred

    # TODO: now add scaled high-frequency components back to the original image
    sharpened_image = image + alpha * high_freq
    return sharpened_image

    # this would just give high frequencies and remove evenything else --> try scaling
    # return high_freq


def evaluate_sharpening(original, blurred, sharpened, blurred_then_sharpened):
    output_dir = './render/part3_7/'
    os.makedirs(output_dir, exist_ok=True)
    # save all the images
    save_image(os.path.join(output_dir, 'original.jpeg'), original)
    save_image(os.path.join(output_dir, 'blurred.jpeg'), blurred)
    save_image(os.path.join(output_dir, 'sharpened.jpeg'), sharpened)
    save_image(os.path.join(output_dir, 'blurred_then_sharpened.jpeg'), blurred_then_sharpened)
    print(f"Images saved in {output_dir}")


if __name__ == '__main__':
    image_number = 16  # TODO: change this to select a different image
    image_path = f"./images/{image_number}.jpeg"
    image = open_image(image_path)  # convert to grayscale if needed but assuming they want color

    # parameters for gaussian filter -- just set to something
    ksize = 20  # kernel size
    sigma = 20.0  # standard deviation

    # create gaussian kernel
    gaussian_kernel = gaussian_filter(ksize, sigma)

    # unsharp mask parameters
    # TODO: IMPORTANT: this the scaling factor for high-frequency components
    alpha = 1.5

    # apply unsharp masking
    sharpened_image = unsharp_mask(image, gaussian_kernel, alpha)

    # create the blurred version for comparison
    blurred_image = convolve_v3(image, gaussian_kernel, mode='same', boundary='symm')

    # this is the blured imaged that lost some hig freq then tried restore
    blurred_then_sharpened_image = unsharp_mask(blurred_image, gaussian_kernel, alpha)

    # evaluate sharpening -- write to folder in function
    evaluate_sharpening(image, blurred_image, sharpened_image, blurred_then_sharpened_image)

    print("Image sharpening completed.")
