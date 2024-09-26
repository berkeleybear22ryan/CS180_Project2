# Image Processing Report

## Overview

This project dives deep into various image processing techniques, blending methods, and visualization approaches using Gaussian and Laplacian pyramids. The objective is to explore, implement, and document methods for creating visually appealing image blends, including advanced blending techniques like multiresolution blending.

The project culminates with practical applications such as tattoo reveal animations and dynamic mask generation for seamless image transitions. The entire codebase has been structured to ensure reproducibility, clarity, and detailed visual documentation.

## Project Structure

### Part 0: Sanity Check - Edge Detection
This part serves as a preliminary verification of the derivative filters' ability to capture image edges accurately. We explore edge detection using simple finite difference operators and visualize the results using binary edge maps.

### Part 1.1: Finite Difference Operators
In this section, we explore the basic x and y derivative filters, manually convolving them with the input image to highlight edges. We document the vertical and horizontal edges separately, followed by a gradient magnitude image and a binary edge map.

### Part 1.2: Derivative of Gaussian (DoG) Filters
We move beyond simple derivative filters and introduce Gaussian smoothing before differentiation. The DoG filters help reduce noise, resulting in a cleaner, more stable edge detection.

### Part 2.1: Unsharp Masking and High-Frequency Enhancement
This part explores unsharp masking for enhancing image sharpness. We compare the effects of different sharpening techniques, with detailed visual comparisons between blurred, sharpened, and original images.

### Part 2.2: Hybrid Images with Color - Bells & Whistles
The goal here is to create hybrid images by blending the low-frequency components of one image with the high-frequency components of another. We experiment with both grayscale and color images, providing detailed visual comparisons for both.

### Part 2.3: Multiresolution Blending
Inspired by the classic Oraple example, we implement multiresolution blending from scratch. The Gaussian and Laplacian pyramids are constructed manually, and various blending techniques are tested, including the creation of blended images using custom masks and pyramids.

### Part 2.4: Blending with Advanced Masking Techniques
We go beyond simple vertical or horizontal seams, generating complex masks for blending images seamlessly. This section showcases a range of masks, from irregular gradients to refined outlines generated using advanced line detection and edge enhancement techniques.

## Final Application: Tattoo Reveal Animation
Using the techniques developed earlier, we create an animation simulating a tattoo reveal on a person’s skin. The animation is built using a sequence of images where the tattoo gradually appears, with additional effects like vertical and horizontal reveals. We then blend the tattoo onto the skin dynamically using the generated masks.

## How to Run the Code

1. **Environment Setup**: Ensure you have all the required Python libraries installed. The project heavily relies on `opencv-python`, `numpy`, `scipy`, and `matplotlib`.

2. **Run Individual Parts**: Each part of the project is modularized into separate scripts (`n1.py`, `n2.py`, etc.). Run these individually to generate the corresponding outputs in the `./render/` directory.

3. **Generate Animations**: For the tattoo reveal animation, ensure that the paths to the images are correct, and run the script. The frames will be saved in the `./tattoo_animation/` folder, which can be compiled into a video or GIF using external tools.

4. **HTML Visualization**: The HTML file `index.html` provides a comprehensive visualization of all results. Simply open this file in a browser to see the images and animations, organized with detailed descriptions.

## Customization

- **Masks**: The project includes functions to generate various types of masks. You can easily add custom masks by providing your own PNG files or modifying the existing mask generation functions.
  
- **Filters**: Select different filters for convolution or define your own in the utility functions. The code is modular, making it straightforward to switch between different filters.

- **Animation**: Modify the animation settings like frame count, speed, and direction in the animation script.

## Conclusion

This project is an extensive exploration of image processing techniques, providing both a theoretical foundation and practical applications. The code is well-documented and organized for easy reproducibility. Feel free to experiment with your own images, masks, and animations to create unique blends and effects.