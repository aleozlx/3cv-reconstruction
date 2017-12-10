import os, sys, glob
import numpy as np
import cv2
from matplotlib import pyplot as plt

# im = cv2.imread('DSC_0551.JPG')
# im = cv2.resize(im, (800,600))
# cv2.imshow('result', im)
# cv2.waitKey(0)
# sys.exit(0)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
pattern = (8,11)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((np.product(pattern),3), np.float32)
objp[:,:2] = np.mgrid[:pattern[0],:pattern[1]].T.reshape(-1,2)
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

if not os.path.exists('checkerboard-opencv'):
    os.system('bash get_datasets.sh')

images = glob.glob('checkerboard/*.JPG')

def _resize(im):
    return cv2.resize(im, tuple(np.array((484,324))*2))

for fname in images:
    img = _resize(cv2.imread(fname) )
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(fname)
    ret, corners = cv2.findChessboardCorners(gray, pattern, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, pattern, corners2, ret)
        # cv2.imshow(fname, img)
        # cv2.waitKey(500)

# cv2.destroyAllWindows()

# NA(3,3)A(1,5)L4(3,1)L4(3,1)
print('0')
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(1)
bold = '\x1B[1m{}\x1B[0m'.format
print(bold('Camera matrix'))
print(mtx)
print(bold('Distortion'))
print(dist)
print(bold('Rotation vectors'))
for v in rvecs:
    print(v)
print(bold('Translation vectors'))
for v in tvecs:
    print(v)

h,w=img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imshow('result', dst)

def homography_cv2(src, dst):
    """ OpenCV implementation as golden standard """
    src_pts = src.reshape(-1,1,2)
    dst_pts = dst.reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, mask.ravel().tolist()

def plot_matching(imA,kpsA,imB,kpsB,matches,H,inliers):
    print('plot matching')
    imB = imB.copy()
    h,w = imA.shape[:2]
    transformed_box = cv2.perspectiveTransform(np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2),H)
    cv2.polylines(imB,[np.int32(transformed_box)],True,255,3, cv2.LINE_AA)
    im = cv2.drawMatches(imA,kpsA,imB,kpsB,matches,None,
        matchColor = (0,255,0), # draw matches in green color
        singlePointColor = None,
        matchesMask = inliers, # draw only inliers
        flags = 2)
    print(im.shape)
    cv2.imshow('aaa',_resize(im))
    # plt.figure(figsize=(20,10))
    # plt.imshow(_resize(im), 'gray')

def step2():
    # Read images
    imA = cv2.cvtColor(cv2.imread("Bear/20171128_IMG_7765.JPG"), cv2.COLOR_BGR2GRAY)
    imB = cv2.cvtColor(cv2.imread("Bear/20171128_IMG_7766.JPG"), cv2.COLOR_BGR2GRAY)
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
    # plot_matching(imA,kpsA,imB,kpsB,matches,H,inliers)
    points1 = np.float32([ kpsA[m.queryIdx].pt for m in matches ])
    points2 = np.float32([ kpsB[m.trainIdx].pt for m in matches ])
    return cv2.findFundamentalMat(points1, points2)

F, F_inliers = step2()
E = np.dot(np.dot(mtx.T, F), mtx)
print("F=",F)
print('E',E)
E_U, E_s, E_V = np.linalg.svd(E)
print('singular values', E_s)

while(1):
    key = cv2.waitKey(33) & 0xFF
    if key == ord('\x1B'):
        sys.exit()

