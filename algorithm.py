import cv2
import numpy as np
from skimage.transform import resize

def adaptive_morphological_opening(image, disk_size, iterations):
    dilated_image = cv2.dilate(image, np.ones((disk_size, disk_size), dtype=np.uint8), iterations=iterations)
    return cv2.erode(dilated_image, np.ones((disk_size, disk_size), dtype=np.uint8), iterations=iterations)

def rgb2yiq(image):
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    i = 0.596 * r - 0.275 * g - 0.321 * b
    q = 0.212 * r - 0.523 * g + 0.311 * b

    return np.stack([y, i, q], axis=-1)

def yiq2rgb(image):
    y = image[:, :, 0]
    i = image[:, :, 1]
    q = image[:, :, 2]

    r = y + 0.956 * i + 0.621 * q
    g = y - 0.272 * i - 0.647 * q
    b = y - 1.107 * i + 1.704 * q

    return np.stack([r, g, b], axis=-1)

def block_truncation_coding(image, sub_block_size=8):
    if image.ndim not in [2, 3]:
        raise ValueError("Input image must be 2D (grayscale) or 3D (RGB)")

    height, width = image.shape[:2]

    if image.ndim == 2:
        quantized_image = np.zeros((height, width))

        for i in range(0, height, sub_block_size):
            for j in range(0, width, sub_block_size):
                block = image[i:i+sub_block_size, j:j+sub_block_size]

                mean_value = np.mean(block)
                variance = np.var(block)

                max_threshold = mean_value + 2 * np.sqrt(variance)

                quantized_block = np.zeros((sub_block_size, sub_block_size))
                quantized_block[block <= mean_value] = 0
                quantized_block[(block > mean_value) & (block <= max_threshold)] = 128
                quantized_block[block > max_threshold] = 255

                quantized_image[i:i+sub_block_size, j:j+sub_block_size] = quantized_block

    elif image.ndim == 3:
        quantized_image = np.zeros((height, width, 3))

        for i in range(0, height, sub_block_size):
            for j in range(0, width, sub_block_size):
                block = image[i:i+sub_block_size, j:j+sub_block_size, :]

                quantized_block = np.zeros((sub_block_size, sub_block_size, 3))

                for k in range(3):
                    mean_value = np.mean(block[:, :, k])
                    variance = np.var(block[:, :, k])

                    max_threshold = mean_value + 2 * np.sqrt(variance)

                    quantized_block[:, :, k][block[:, :, k] <= mean_value] = 0
                    quantized_block[:, :, k][(block[:, :, k] > mean_value) & (block[:, :, k] <= max_threshold)] = 128
                    quantized_block[:, :, k][block[:, :, k] > max_threshold] = 255

                quantized_image[i:i+sub_block_size, j:j+sub_block_size, :] = quantized_block

    return quantized_image

def refine_compressed_image(compressed_image):
    compressed_image_uint8 = np.array(compressed_image, dtype=np.uint8)
    refined_image = cv2.GaussianBlur(compressed_image_uint8, (3, 3), 0)
    return refined_image

def color_image_compression(image_path, sub_block_size=4, k=2):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    original_image_blurred = cv2.GaussianBlur(original_image, (3, 3), 0)

    yiq_image = rgb2yiq(original_image_blurred)
    luminance = yiq_image[:, :, 0]

    smoothed_luminance = adaptive_morphological_opening(luminance, disk_size=3, iterations=1)
    downsampled_luminance = resize(smoothed_luminance, (luminance.shape[0] // 2, luminance.shape[1] // 2), anti_aliasing=True)
    quantized_luminance = block_truncation_coding(downsampled_luminance, sub_block_size)

    filtered_chrominance_i = resize(yiq_image[:, :, 1], (luminance.shape[0] // 2, luminance.shape[1] // 2), anti_aliasing=True)
    filtered_chrominance_q = resize(yiq_image[:, :, 2], (luminance.shape[0] // 2, luminance.shape[1] // 2), anti_aliasing=True)
    quantized_chrominance_i = block_truncation_coding(filtered_chrominance_i, sub_block_size)
    quantized_chrominance_q = block_truncation_coding(filtered_chrominance_q, sub_block_size)

    reconstructed_luminance = resize(quantized_luminance, luminance.shape, anti_aliasing=True)
    reconstructed_chrominance_i = resize(quantized_chrominance_i, yiq_image[:, :, 1].shape, anti_aliasing=True)
    reconstructed_chrominance_q = resize(quantized_chrominance_q, yiq_image[:, :, 2].shape, anti_aliasing=True)

    compressed_yiq = np.stack([reconstructed_luminance, reconstructed_chrominance_i, reconstructed_chrominance_q], axis=-1)
    compressed_image = yiq2rgb(compressed_yiq)

    compressed_image = refine_compressed_image(compressed_image)

    return original_image, reconstructed_luminance, compressed_image

# Uncomment the following lines if you want to test the algorithm independently
# image_path = 'path/to/your/image.jpg'
# original_image, reconstructed_luminance, compressed_image = color_image_compression(image_path)
# plt.imshow(original_image)
# plt.show()
