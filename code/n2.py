from utility import *



if __name__ == '__main__':
    # TODO: all my images should be between 0 and 255 and are 3d no matter if black and white
    image_number = 16
    image_path = f"./images/{image_number}.jpeg"
    output_path = "./render/part0_1/"




    image = open_image(image_path)
    print(f"image.shape: {image.shape}")
    print(image.max(), image.min())



    imgg = convolve_v1(image[:, :, 0], G, mode='same', boundary='symm')
    imgx = convolve_v1(imgg, D_x, mode='same', boundary='symm')
    imgy = convolve_v1(imgg, D_y, mode='same', boundary='symm')
    imgxy = combine_images_gradient_magnitude(imgx, imgy)
    imgxy_bin = threshold_image(imgxy, 10)


    # shorter way
    c1 = convolve_v1(G, D_x, mode="full", boundary='fill')
    c2 = convolve_v1(G, D_y, mode="full", boundary='fill')
    # c3 = convolve_v1(c1, c2)
    imgx_s = convolve_v1(image[:, :, 0], c1, mode='same', boundary='symm')
    imgy_s = convolve_v1(image[:, :, 0], c2, mode='same', boundary='symm')
    imgxy_s = combine_images_gradient_magnitude(imgx_s, imgy_s)
    imgxy_s_bin = threshold_image(imgxy_s, 10)


    save_image(output_path + "1_imgg.jpeg", imgg)
    save_image(output_path + "2_imgx.jpeg", imgx)
    save_image(output_path + "3_imgy.jpeg", imgy)
    save_image(output_path + "4_imgxy.jpeg", imgxy)
    save_image(output_path + "5_imgxy_s.jpeg", imgxy_s)

    save_image(output_path + "6_imgxy_bin.jpeg", imgxy_bin)
    save_image(output_path + "7_imgxy_s_bin.jpeg", imgxy_s_bin)



