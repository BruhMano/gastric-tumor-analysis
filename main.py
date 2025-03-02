import os
import pydicom
import numpy as np
import cv2

def get_marked_area(filename: str, save_file_name: str):
    ds = pydicom.dcmread(filename)
    image = (np.maximum(ds.pixel_array,0)/ds.pixel_array.max())*255
    print(image)
    overlay = ds.overlay_array(0x6000)*255
    # print(ds.get((0x0029,0x1094)))
    contours, _ = cv2.findContours(overlay, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key= cv2.contourArea)
    # tr_img = cv2.drawContours(image, max_contour, -1, (255,0,0), 1)
    # cv2.imwrite(save_file_name, overlay)
    cv2.imwrite(save_file_name, tr_img)
# (0029,1094)

def main():
    get_marked_area("Resulting View/00002_1.3.12.2.1107.5.8.15.133379.30000025020205011643700000096.dcm","overlay.png")
    # i = 0
    # for filename in os.listdir("Resulting View"):
    #     get_marked_area(f"Resulting View/{filename}",f"masks/mask{i}.png")
    #     i+=1
    i = 0
    for filename in os.listdir("Resulting View"):
        get_marked_area(f"Resulting View/{filename}",f"results/frame{i}.png")
        i+=1

if __name__ == "__main__":
    main()