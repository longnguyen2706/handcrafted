import cv2
import numpy as np
import matplotlib.pyplot as plt

def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

def gen_sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    # kp is the keypoints
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

def show_sift_features(gray_img, color_img, kp):
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))

def show_rgb_img(img):
    """Convenience function to display a typical color image"""
    return plt.imshow(cv2.cvtColor(img, cv2.CV_32S))

def explain_keypoint(kp):
    print ('angle\n', kp.angle)
    print ('\nclass_id\n', kp.class_id)
    print ('\noctave (image scale where feature is strongest)\n', kp.octave)
    print ('\npt (x,y)\n', kp.pt)
    print ('\nresponse\n', kp.response)
    print ('\nsize\n', kp.size)

def main():
   img = cv2.imread('/home/long/Desktop/gray.jpg')
   gray = to_gray(img)
   kp, desc = gen_sift_features(gray)
   print (kp)
   print(desc)
   show_sift_features(gray, img, kp)
   explain_keypoint(kp[16])

if __name__ == "__main__":
    main()