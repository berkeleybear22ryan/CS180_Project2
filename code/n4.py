import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage
import skimage.transform as sktr
from tqdm import tqdm
import math
import os
import cv2

# make sure to create output directory if it does not exist
output_dir = './render/part4_5/'
os.makedirs(output_dir, exist_ok=True)


# this is the alignment functions
def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)



# recenter adjust if needed later
def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int)(np.abs(2 * r + 1 - R))
    cpad = (int)(np.abs(2 * c + 1 - C))
    return np.pad(
        im, [(0 if r > (R - 1) / 2 else rpad, 0 if r < (R - 1) / 2 else rpad),
             (0 if c > (C - 1) / 2 else cpad, 0 if c < (C - 1) / 2 else cpad),
             (0, 0)], 'constant')


def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy


def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape

    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2


def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
    len2 = np.sqrt((p4[1] - p3[1]) ** 2 + (p4[0] - p3[0]) ** 2)
    dscale = len2 / len1
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale, channel_axis=2, mode='constant', anti_aliasing=True)
    else:
        im2 = sktr.rescale(im2, 1. / dscale, channel_axis=2, mode='constant', anti_aliasing=True)
    return im1, im2


def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta * 180 / np.pi)
    return im1, dtheta


def match_img_size(im1, im2):
    # note: make images the same size -- check
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    if h1 < h2:
        im2 = im2[int(np.floor((h2 - h1) / 2.)): -int(np.ceil((h2 - h1) / 2.)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1 - h2) / 2.)): -int(np.ceil((h1 - h2) / 2.)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2 - w1) / 2.)): -int(np.ceil((w2 - w1) / 2.)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1 - w2) / 2.)): -int(np.ceil((w1 - w2) / 2.)), :]
    assert im1.shape == im2.shape
    return im1, im2


def align_images(im1, im2):
    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2, angle


# hybrid image functions below ...
def gaussian_filter(shape, sigma):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def fft_convolve2d(x, y):
    fr = np.fft.fft2(x)
    fr2 = np.fft.fft2(y, fr.shape)
    m, n = fr.shape
    cc = np.real(np.fft.ifft2(fr * fr2))
    cc = np.roll(cc, -m // 2 + 1, axis=0)
    cc = np.roll(cc, -n // 2 + 1, axis=1)
    return cc


def hybrid_image(im1, im2, sigma1, sigma2, blend_ratio=0.5):
    print("Creating hybrid image...")

    assert im1.shape == im2.shape, "Images must have the same shape"

    hybrid = np.zeros_like(im1)
    for channel in range(3):
        print(f"Processing channel {channel + 1}/3...")
        gf1 = gaussian_filter(im1[:, :, channel].shape, sigma1)
        gf2 = gaussian_filter(im2[:, :, channel].shape, sigma2)

        im1_low = fft_convolve2d(im1[:, :, channel], gf1)
        im2_low = fft_convolve2d(im2[:, :, channel], gf2)
        im2_high = im2[:, :, channel] - im2_low

        hybrid[:, :, channel] = blend_ratio * im1_low + (1 - blend_ratio) * im2_high

    print("Normalizing the result...")
    hybrid = np.clip(hybrid, 0, 1)

    print("Hybrid image created.")
    return hybrid


def rotate_image(image, angle):
    return sktr.rotate(image, angle, resize=False, mode='reflect', preserve_range=True)


def visualize_scale_space(image, n_scales=5):
    print("Visualizing scale space...")
    scales = [2 ** i for i in range(n_scales)]
    fig, axes = plt.subplots(1, n_scales, figsize=(15, 3))
    for i, scale in tqdm(enumerate(scales), total=n_scales, desc="Processing scales"):
        sigma = scale
        smoothed = np.dstack([ndimage.gaussian_filter(image[:, :, c], sigma=sigma) for c in range(3)])
        axes[i].imshow(np.clip(smoothed, 0, 1))
        axes[i].set_title(f'σ = {sigma}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scale_space.png'))
    plt.close()
    print("Scale space visualization complete.")


# update with TQDM ...
def build_gaussian_pyramid(image, max_levels=5):
    print("Building Gaussian pyramid...")
    pyramid = [image]
    for i in tqdm(range(max_levels - 1), desc="Building levels"):
        image = np.dstack([ndimage.gaussian_filter(image[:, :, c], sigma=2) for c in range(3)])
        image = image[::2, ::2]
        pyramid.append(image)
    print("Gaussian pyramid built.")
    return pyramid


def build_laplacian_pyramid(gaussian_pyramid):
    print("Building Laplacian pyramid...")
    laplacian_pyramid = []
    for i in tqdm(range(len(gaussian_pyramid) - 1), desc="Building levels"):
        size = gaussian_pyramid[i].shape[:2]
        expanded = np.zeros_like(gaussian_pyramid[i])
        for c in range(3):
            expanded[:, :, c] = ndimage.zoom(gaussian_pyramid[i + 1][:, :, c], 2)[:size[0], :size[1]]
        laplacian = gaussian_pyramid[i] - expanded
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    print("Laplacian pyramid built.")
    return laplacian_pyramid


def display_pyramid(pyramid, title):
    print(f"Displaying {title}...")
    n = len(pyramid)
    fig, axes = plt.subplots(1, n, figsize=(15, 3))
    for i, img in tqdm(enumerate(pyramid), total=n, desc="Displaying levels"):
        axes[i].imshow(np.clip(img, 0, 1))
        axes[i].set_title(f'Level {i}')
        axes[i].axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title.replace(" ", "_")}.png'))
    plt.close()
    print(f"{title} displayed and saved.")


def visualize_frequency_domain(image, title):
    if image.ndim == 3:
        image = np.mean(image, axis=2)  # convert to grayscale ... check if needed
    ft = np.fft.fft2(image)
    ft_shift = np.fft.fftshift(ft)
    magnitude_spectrum = np.log(np.abs(ft_shift) + 1)

    plt.figure(figsize=(10, 5))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title.replace(" ", "_")}.png'))
    plt.close()


def display_frequency_analysis(image, title):
    print(f"Performing frequency analysis for {title}...")
    gray_image = np.mean(image, axis=2)  # convert to grayscale ... check if needed
    ft = np.fft.fft2(gray_image)
    ft_shift = np.fft.fftshift(ft)
    magnitude_spectrum = np.log(np.abs(ft_shift) + 1)

    plt.figure(figsize=(10, 5))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(f'Frequency Analysis: {title}')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'frequency_analysis_{title.replace(" ", "_")}.png'))
    plt.close()
    print(f"Frequency analysis for {title} complete and saved.")



if __name__ == "__main__":
    print("Starting hybrid image creation process...")

    # load images ...
    print("Loading images...")
    im1 = plt.imread('./images/24.jpeg') / 255.  # high sf, man
    im2 = plt.imread('./images/22.jpeg') / 255  # low sf, cat
    print("Images loaded.")

    # save original images
    plt.imsave(os.path.join(output_dir, 'original_image1.png'), im1)
    plt.imsave(os.path.join(output_dir, 'original_image2.png'), im2)

    # frequency analysis of input images
    visualize_frequency_domain(im1, "Input Image 1 FFT")
    visualize_frequency_domain(im2, "Input Image 2 FFT")

    # align images
    print("Aligning images...")
    print("Please select 2 points in each image for alignment.")
    im1_aligned, im2_aligned, rotation_angle = align_images(im1, im2)
    print("Images aligned.")

    # save aligned images
    plt.imsave(os.path.join(output_dir, 'aligned_image1.png'), im1_aligned)
    plt.imsave(os.path.join(output_dir, 'aligned_image2.png'), im2_aligned)

    # create hybrid image ... default way almost all but the text used this
    # 17 + 18
    sigma1 = 20  # TODO: need to adjust this value to change the cutoff for high frequencies
    sigma2 = 10  # TODO: need to adjust this value to change the cutoff for low frequencies
    blend_ratio = 0.5  # TODO: need to adjust this to control the blend (higher values emphasize im1)

    # # OTHERS ... just text
    # sigma1 = 10  # TODO: need to adjust this value to change the cutoff for high frequencies
    # sigma2 = 3  # TODO: need to adjust this value to change the cutoff for low frequencies
    # blend_ratio = 0.5  # TODO: need to adjust this to control the blend (higher values emphasize im1)

    # apply filters and visualize
    gf1 = gaussian_filter(im1_aligned.shape[:2], sigma1)
    gf2 = gaussian_filter(im2_aligned.shape[:2], sigma2)

    im1_low = np.dstack([fft_convolve2d(im1_aligned[:, :, c], gf1) for c in range(3)])
    im2_low = np.dstack([fft_convolve2d(im2_aligned[:, :, c], gf2) for c in range(3)])
    im2_high = im2_aligned - im2_low

    visualize_frequency_domain(im1_low, "Low Frequency Image FFT Filtered")
    visualize_frequency_domain(im2_high, "High Frequency Image FFT Filtered")

    hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2, blend_ratio)

    # rotate hybrid image back to original orientation
    print("Rotating hybrid image to original orientation...")
    hybrid_rotated = rotate_image(hybrid, -rotation_angle * 180 / np.pi)

    # save hybrid images
    plt.imsave(os.path.join(output_dir, 'hybrid_image_aligned.png'), hybrid)
    plt.imsave(os.path.join(output_dir, 'hybrid_image_rotated.png'), hybrid_rotated)

    # plus save bw image -- because think it will look better
    hybrid_rotated_uint8 = cv2.normalize(hybrid_rotated, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    gray_image = cv2.cvtColor(hybrid_rotated_uint8, cv2.COLOR_BGR2GRAY)

    plt.imsave(os.path.join(output_dir, 'hybrid_image_rotated_bw.png'),gray_image, cmap="gray")

    # frequency analysis of hybrid image
    visualize_frequency_domain(hybrid_rotated, "Hybrid Image FFT")

    # display hybrid images
    print("Displaying hybrid images...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.imshow(hybrid)
    ax1.set_title('Hybrid Image (Aligned)')
    ax1.axis('off')
    ax2.imshow(hybrid_rotated)
    ax2.set_title('Hybrid Image (Original Orientation)')
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hybrid_images_comparison.png'))
    plt.close()
    print("Hybrid images displayed and saved.")

    # visualize scale space
    visualize_scale_space(hybrid_rotated)

    # compute and display gaussian and laplacian Pyramids
    gaussian_pyramid = build_gaussian_pyramid(hybrid_rotated)
    laplacian_pyramid = build_laplacian_pyramid(gaussian_pyramid)

    display_pyramid(gaussian_pyramid, 'Gaussian Pyramid')
    display_pyramid(laplacian_pyramid, 'Laplacian Pyramid')

    # frequency analysis
    display_frequency_analysis(im1, 'Input Image 1')
    display_frequency_analysis(im2, 'Input Image 2')
    display_frequency_analysis(hybrid_rotated, 'Hybrid Image')

    print("Hybrid image creation process complete.")
