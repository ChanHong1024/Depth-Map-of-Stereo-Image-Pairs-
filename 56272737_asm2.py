# #####################
# CS4186 ASM2
# CHAN WAI HONG 
# 56272737
# #####################

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

paths = ["Art\\", "Reindeer\\", "Dolls\\"]

# #######################
#  Feature Extracting / Image Retification
# #######################

def image_rectification(left_view, right_view, currentfname):  # i = index of loop
        
    good, pts1, pts2 = [], [], []

    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(left_view, None)
    kp2, des2 = orb.detectAndCompute(right_view, None)

    # kp from ORB
    left_view_kp = cv.drawKeypoints(left_view, kp1, outImage=np.array([]), color=(0, 0, 255))
    cv.imwrite("left_view_kp_"+ currentfname +".jpg",left_view_kp)

    right_view_kp =cv.drawKeypoints(right_view, kp2, outImage=np.array([]), color=(0, 0, 255))
    cv.imwrite("right_view_kp_"+ currentfname +".jpg",right_view_kp)

    # FLANN matches
    index_params = dict(algorithm=0, trees=100)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    flann_match_pairs = flann.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)

    # remove false matches
    threshold = 0.3
    threshold = 1-threshold
    for i,(m,n) in enumerate(flann_match_pairs):
        if m.distance < threshold * n.distance:
            good.append([m])
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    #print("good matches: " + str(len(good)))

    # compute Fundamental Matrix
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    fMatrix, inliers = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)

    # plot out all matches
    matches_img = cv.drawMatchesKnn(left_view,kp1,right_view,kp2,good,None,flags=2)
    plt.figure(figsize = (200,10))
    plt.imshow(matches_img)
    cv.imwrite("matches_"+ currentfname +".jpg", matches_img)

    # rectification transform
    h1, w1 = left_view.shape
    h2, w2 = right_view.shape
    thresh = 0
    _, H1, H2 = cv.stereoRectifyUncalibrated(
        np.float32(pts1), 
        np.float32(pts2), 
        fMatrix, imgSize=(w1, h1), 
        threshold=thresh,)

    # undistort
    left_view_undistorted = cv.warpPerspective(left_view, H1, (w1, h1))
    right_view_undistorted = cv.warpPerspective(right_view, H2, (w2, h2))
    cv.imwrite("left_view_undistorted_"+ currentfname +".png", left_view_undistorted)
    cv.imwrite("right_view_undistorted_"+ currentfname +".png", right_view_undistorted)

    return left_view_undistorted, right_view_undistorted

# #######################
#  Stereo Disparities Processing
# #######################

for i in range(3):

    currentfname = paths[i][:-1].lower()

    left_view = cv.imread(paths[i] + 'view1.png', cv.IMREAD_GRAYSCALE)
    right_view = cv.imread(paths[i] + 'view5.png', cv.IMREAD_GRAYSCALE)

    left_view_undistorted, right_view_undistorted = image_rectification(left_view,right_view, currentfname)

    # Semi-Global Block Matching
    stereo = cv.StereoSGBM_create()
    # parameters that I think is THE BEST
    stereo.setMinDisparity(-128)
    stereo.setNumDisparities(256)      
    stereo.setBlockSize(16)                       
    stereo.setDisp12MaxDiff(0)                       
    stereo.setUniquenessRatio(5)
    stereo.setSpeckleRange(2)
    stereo.setSpeckleWindowSize(200)

    # compute disparities
    disparity = stereo.compute(left_view_undistorted, right_view_undistorted)
    # reduce the depth of whole image
    disparity = cv.normalize(disparity, disparity, alpha=255, beta=0, norm_type=cv.NORM_MINMAX)
    # store image as 256 color format
    disparity = np.uint8(disparity)


    cv.imwrite("result_"+currentfname+".png", disparity)

    from PIL import Image
    import numpy
    import math

    def psnr(img1, img2):
        mse = numpy.mean( ((img1 - img2)) ** 2 )
        if mse == 0:
            return 'INF'
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


    gt = cv.imread(paths[i] + 'disp1.png', cv.IMREAD_GRAYSCALE)
    pre = cv.imread("result_"+currentfname+".png", cv.IMREAD_GRAYSCALE)
    print("psnr of image " +currentfname+" :" + str(psnr(disparity, gt)))