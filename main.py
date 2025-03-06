import os
import pydicom
import numpy as np
import cv2

def get_image(filename: str, save_file_name: str):
    ds = pydicom.dcmread(filename)
    image = (np.maximum(ds.pixel_array, 0) / ds.pixel_array.max()) * 255
    cv2.imwrite(save_file_name,image)

def get_marked_area_mask(filename: str, save_file_name: str):
    ds = pydicom.dcmread(filename)
    image = (np.maximum(ds.pixel_array,0)/ds.pixel_array.max())*255
    overlay = ds.overlay_array(0x6000)*255
    contours, _ = cv2.findContours(overlay, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key= cv2.contourArea)
    mask = cv2.drawContours(np.zeros((512,512)), [max_contour], -1, (255,255,255), thickness=-1)
    # cv2.imwrite(save_file_name, overlay)
    cv2.imwrite(save_file_name, mask)
# (0029,1094)

def get_masked_image(maskfile: str, imagefile: str, resultpath: str):
    mask = cv2.imread(maskfile).astype(dtype=np.uint8)
    image = cv2.imread(imagefile)
    masked_image = cv2.bitwise_and(image,image,mask=mask[:,:,2])
    cv2.imwrite(resultpath,masked_image)

def main():
    i = 0
    for filename in os.listdir("Resulting View"):
        get_image(f"Resulting View/{filename}",f"images/frame{i}.png")
        get_marked_area_mask(f"Resulting View/{filename}",f"masks/frame{i}.png")
        get_masked_image(f"masks/frame{i}.png", f"images/frame{i}.png", f"masked/frame{i}.png")
        i+=1

if __name__ == "__main__":
    main()