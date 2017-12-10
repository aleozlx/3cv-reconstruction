import os, sys, glob, pickle
import numpy as np
import cv2
from matplotlib import pyplot as plt
import get_datasets
get_datasets.run()

def sresize(im, s):
    return cv2.resize(im, tuple(np.round(np.flip(im.shape[:2], 0) * s).astype(int)))

def mem(key, obj = None):
    fname = '{}.pkl'.format(key)
    if mem_exists(obj):
        with open(fname, 'wb') as f:
            pickle.dump(obj, f)
    elif os.path.exists(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)

def mem_exists(obj):
    return obj is not None

def step1():
    print('Camera calibration...')
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    pattern = (8,11)
    objp = np.zeros((np.product(pattern),3), np.float32)
    objp[:,:2] = np.mgrid[:pattern[0],:pattern[1]].T.reshape(-1,2)
    objpoints = []
    imgpoints = []

    _cache = mem('calibrateCamera')
    if mem_exists(_cache):
        ret, mtx, dist, rvecs, tvecs = _cache
        img = mem('calibrateCamera_img')
    else:
        for fname in glob.glob('checkerboard/*.JPG'):
            img = sresize(cv2.imread(fname), 2/7)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # print(fname)
            ret, corners = cv2.findChessboardCorners(gray, pattern, None)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)
                img = cv2.drawChessboardCorners(img, pattern, corners2, ret)
                # cv2.imshow(fname, img)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        mem('calibrateCamera', (ret, mtx, dist, rvecs, tvecs))
        mem('calibrateCamera_img', img)

    _cache = mem('newCameraMatrix')
    if mem_exists(_cache):
        newcameramtx = _cache
    else:
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        mem('newCameraMatrix', newcameramtx)
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    cv2.imshow('camera-calibration', dst)
    return mtx, newcameramtx

_K, K = step1()
print('Camera matrix:')
print(K)

def step2():
    def homography_cv2(src, dst):
        src_pts = src.reshape(-1,1,2)
        dst_pts = dst.reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H, mask.ravel().tolist()

    def plot_matching(imA,kpsA,imB,kpsB,matches,H,inliers):
        print('plot matching')
        imB = imB.copy()
        h,w = imA.shape[:2]
        transformed_box = cv2.perspectiveTransform(np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2),H)
        cv2.polylines(imB,[np.int32(transformed_box)],True,(255,255,255),3, cv2.LINE_AA)
        im = cv2.drawMatches(imA,kpsA,imB,kpsB,matches,None,
            matchColor = (255,255,0), # draw matches in green color
            singlePointColor = None,
            matchesMask = inliers, # draw only inliers
            flags = 2)
        print(im.shape)
        cv2.imshow('feature-matching', im)

    imA = sresize(cv2.imread("Deer/20171128_IMG_7754.JPG"), 0.2)
    imB = sresize(cv2.imread("Deer/20171128_IMG_7755.JPG"), 0.2)
    # SIFT+FLANN matching
    sift = cv2.xfeatures2d.SIFT_create()
    (kpsA, desA) = sift.detectAndCompute(imA, None)
    (kpsB, desB) = sift.detectAndCompute(imB, None)
    flann = cv2.FlannBasedMatcher(dict(algorithm = 0, trees = 5), dict(checks = 50))
    # Lowe's ratio test
    matches = [m for m,n in flann.knnMatch(desA,desB,k=2) if m.distance < 0.7*n.distance] 
    print(matches)
    if len(matches)>10:
        while 1: # Just try again when getting singular matrix
            try:
                H, inliers = homography_cv2(
                    np.float32([ kpsA[m.queryIdx].pt for m in matches ]),
                    np.float32([ kpsB[m.trainIdx].pt for m in matches ]))
                break
            except np.linalg.linalg.LinAlgError:
                continue
    else:
        print("Not enough matches are found")
        sys.exit(1)
    print(H, len(inliers))
    plot_matching(imA,kpsA,imB,kpsB,matches,H,inliers)
    points1 = np.float32([ kpsA[m.queryIdx].pt for m in matches ])
    points2 = np.float32([ kpsB[m.trainIdx].pt for m in matches ])
    F, F_inliers = cv2.findFundamentalMat(points1, points2)
    return np.dot(np.dot(K.T, F), K)

E = step2()
print('Essential matrix')
print(E)
print('singular values', np.linalg.svd(E)[1])

while(1):
    key = cv2.waitKey(33) & 0xFF
    if key == ord('\x1B'):
        sys.exit()

