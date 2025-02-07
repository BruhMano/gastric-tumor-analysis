import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_marked_area(filename: str):
    ds = pydicom.dcmread(filename)
    image = (np.maximum(ds.pixel_array,0)/ds.pixel_array.max())*255
    print(image)
    overlay = ds.overlay_array(0x6000)*255
    contours, _ = cv2.findContours(overlay, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    tr_img = cv2.drawContours(image, contours, -1, (255,0,0), 1)
    cv2.imwrite("overlay.png", overlay)
    cv2.imwrite("image.png", tr_img)
    # contours_array = np.array(ds.ContourData)
    # plt.imshow(image, cmap='gray')
    # # plt.plot(contours_array[:, 0], contours_array[:, 1], color='red')  # Предполагается, что это 2D-контуры
    # # plt.title('Image with Contours')
    # plt.show()

def main():
    get_marked_area("Resulting View/00030_1.3.12.2.1107.5.8.15.133379.30000025020205011643700000308.dcm")

if __name__ == "__main__":
    main()