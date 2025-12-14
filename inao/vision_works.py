# -*- encoding: UTF-8 -*-
# Get 5 images from NAO, save them incrementally in a folder, then close.

import sys
import time
import cv2  # OpenCV for image processing and saving
import numpy as np  # For array handling
import os  # For directory creation

from naoqi import ALProxy


def captureMultipleImages(IP, PORT, num_images=5):
    """
    Capture multiple images from NAO, save them incrementally in a folder.
    """

    camProxy = ALProxy("ALVideoDevice", IP, PORT)
    resolution = 2  # VGA
    colorSpace = 11  # RGB

    # Subscribe once for the session
    videoClient = camProxy.subscribe("python_client", resolution, colorSpace, 5)

    # Create output directory if it doesn't exist
    output_dir = "./captured_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Capturing {} images...".format(num_images))

    for i in range(num_images):
        t0 = time.time()

        # Get a camera image.
        # image[6] contains the image data passed as an array of ASCII chars.
        naoImage = camProxy.getImageRemote(videoClient)

        t1 = time.time()

        # Time the image transfer.
        print("Image {} acquisition delay: {:.3f} seconds".format(i, t1 - t0))

        # Get the image size and pixel array.
        imageWidth = naoImage[0]
        imageHeight = naoImage[1]
        array = naoImage[6]

        # Convert the array to a numpy array and reshape to image dimensions
        # colorSpace=11 is RGB, so 3 channels
        image_array = np.fromstring(array, dtype=np.uint8).reshape(
            (imageHeight, imageWidth, 3)
        )

        # OpenCV uses BGR by default, but NAO provides RGB, so convert RGB to BGR for proper saving
        bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Save the image with incremental index
        filename = os.path.join(output_dir, "image_{}.png".format(i))
        cv2.imwrite(filename, bgr_image)
        print("Saved: {}".format(filename))

        # Optional: Display each image (comment out if not needed)
        # cv2.imshow("NAO Camera Image {}".format(i), bgr_image)
        # cv2.waitKey(500)  # Show for 500ms
        # cv2.destroyWindow("NAO Camera Image {}".format(i))

        # Small delay between captures
        time.sleep(0.5)

    # Unsubscribe after all captures
    camProxy.unsubscribe(videoClient)

    print("All {} images captured and saved in: {}".format(num_images, output_dir))
    print("Program closing.")


if __name__ == "__main__":
    IP = "192.168.1.118"  # Replace here with your NaoQi's IP address.
    PORT = 9559

    # Read IP address from first argument if any.
    if len(sys.argv) > 1:
        IP = sys.argv[1]

    # Capture 5 images
    captureMultipleImages(IP, PORT, num_images=5)
