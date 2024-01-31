# # read image
# image = cv2.imread('example.jpg', cv2.IMREAD_COLOR)
# image = image[1:1440:20,1:1440:20,:]
# RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(1)
# plt.imshow(RGB_img)

# print("Hello")
# x0 = bilateral(predIllumNumpy[:,:,0], 20, 200)
# print("x1 done")
# x1 = bilateral(RGB_img[:,:,1], 20, 200)
# print("x2 done")
# x2 = bilateral(RGB_img[:,:,2], 20, 200)
# fully_processed = cv2.merge([x0,x1,x2])
# print("Done")
# normalizedImg = np.zeros((1440, 1440, 3))
# normalizedImg = cv2.normalize(fully_processed,  normalizedImg, 0, 255, cv2.NORM_MINMAX)

# plt.figure(2)
# plt.imshow(normalizedImg.astype(int))


# import cv2
# import numpy as np

# # Assuming you have an image with shape (3, 256, 256)
# image = np.random.randint(0, 256, size=(3, 256, 256), dtype=np.uint8)

# predIllumNumpy = pred_illum.cpu().detach().numpy()
# gtIllumNumpy = gt_illum.cpu().detach().numpy()
# # Transpose the image to (256, 256, 3)
# predIllum_np = np.transpose(predIllumNumpy[0,:,:,:], (1, 2, 0))
# gtIllum_np = np.transpose(gtIllumNumpy[0,:,:,:], (1, 2, 0))
# # Apply bilateral filter to each channel independently
# bilateral_filtered_image = np.zeros_like(predIllum_np)

# for i in range(3):
#     bilateral_filtered_image[:, :, i] = cv2.bilateralFilter(predIllum_np[:, :, i], d=9, sigmaColor=75, sigmaSpace=75)

# # Display the original and filtered images
# cv2.imshow('Original Image', predIllum_np)
# cv2.imshow('Bilateral Filtered Image', bilateral_filtered_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# normalizedImg = cv2.normalize(fully_processed,  normalizedImg, 0, 255, cv2.NORM_MINMAX)

# file_name = f'bilateral_filtered_image.png'
# cv2.imwrite(file_name, cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_RGB2BGR))

# cv2.imwrite(f'predImage.png', cv2.cvtColor(predIllum_np, cv2.COLOR_RGB2BGR))

def mse(image1, image2):
    # Ensure the images have the same shape
    assert image1.shape == image2.shape, "Input images must have the same shape"

    # Calculate MSE for each channel
    mse_channels = np.mean((image1 - image2) ** 2, axis=(0, 1))

    # Calculate the overall MSE by averaging the channel-wise MSE values
    mse_total = np.mean(mse_channels)

    return mse_total

def applyBF(pred_illum,gt_illum):
    predIllumNumpy = pred_illum.cpu().detach().numpy()
    gtIllumNumpy = gt_illum.cpu().detach().numpy()
    # Transpose the image to (256, 256, 3)
    predIllum_np = np.transpose(predIllumNumpy[0,:,:,:], (1, 2, 0))
    gtIllum_np = np.transpose(gtIllumNumpy[0,:,:,:], (1, 2, 0))
    # Apply bilateral filter to each channel independently
    bilateral_filtered_image = np.zeros_like(predIllum_np)

    for i in range(3):
        bilateral_filtered_image[:, :, i] = cv2.bilateralFilter(predIllum_np[:, :, i], d=9, sigmaColor=75, sigmaSpace=75)
    # return bilateral_filtered_image

    mseResult = mse(gtIllum_np,bilateral_filtered_image)
    mseResult2 = mse(gtIllum_np,predIllum_np)
    mseResult3 = mse(predIllum_np,bilateral_filtered_image)

    print(f"mse between gt and pred: {mseResult2}")
    print(f"mse between gt and bilateral: {mseResult}")
    print(f"mse between pred and bilateral: {mseResult3}")
    # return mseResult