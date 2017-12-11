import os, sys, glob, pickle, itertools
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
        imB = imB.copy()
        h,w = imA.shape[:2]
        transformed_box = cv2.perspectiveTransform(np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2),H)
        cv2.polylines(imB,[np.int32(transformed_box)],True,(255,255,255),3, cv2.LINE_AA)
        im = cv2.drawMatches(imA,kpsA,imB,kpsB,matches,None,
            matchColor = (255,255,0),
            singlePointColor = None,
            matchesMask = inliers,
            flags = 2)
        cv2.imshow('feature-matching', im)

        def more_points(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONUP:
                if x<=imA.shape[1]:
                    print('points1.append(({}, {}))'.format(x, y))
                else:
                    print('points2.append(({}, {}))'.format(x-imA.shape[1], y))
        cv2.setMouseCallback('feature-matching', more_points)

    imA = sresize(cv2.imread("Deer/20171128_IMG_7754.JPG"), 0.2)
    imB = sresize(cv2.imread("Deer/20171128_IMG_7755.JPG"), 0.2)
    # SIFT+FLANN matching
    sift = cv2.xfeatures2d.SIFT_create()
    (kpsA, desA) = sift.detectAndCompute(imA, None)
    (kpsB, desB) = sift.detectAndCompute(imB, None)
    flann = cv2.FlannBasedMatcher(dict(algorithm = 0, trees = 5), dict(checks = 50))
    # Lowe's ratio test
    matches = [m for m,n in flann.knnMatch(desA,desB,k=2) if m.distance < 0.7*n.distance] 
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
    print('Homography:')
    print(H)
    plot_matching(imA,kpsA,imB,kpsB,matches,H,inliers)
    print('H inliers:', np.sum(inliers))
    inliers = np.flatnonzero(np.array(inliers))
    points1 = np.float32([ kpsA[m.queryIdx].pt for m in matches ])[inliers]
    points2 = np.float32([ kpsB[m.trainIdx].pt for m in matches ])[inliers]
    F, F_inliers = cv2.findFundamentalMat(points1, points2)
    F_inliers = np.flatnonzero(F_inliers)
    return F, np.dot(np.dot(K.T, F), K), points1[F_inliers].tolist(), points2[F_inliers].tolist()

F, E, points1, points2 = step2()
if 1:
    points1.append([441, 168])
    points2.append([516, 207])
    points1.append([360, 231])
    points2.append([426, 240])
    points1.append([390, 209])
    points2.append([457, 231])
    points1.append([411, 202])
    points2.append([479, 229])
    points1.append([259, 185])
    points2.append([351, 174])
    points1.append([287, 169])
    points2.append([387, 169])
    points1.append([276, 241])
    points2.append([352, 227])
    points1.append([285, 244])
    points2.append([362, 233])
    points1.append([309, 270])
    points2.append([372, 260])
    points1.append([268, 341])
    points2.append([342, 311])
    points2.append([431, 306])
    points1.append([351, 302])
    points1.append([382, 324])
    points2.append([440, 333])
    points1.append([466, 291])
    points2.append([542, 332])

for a,b in zip(points1, points2):
    a = np.array(a+[1])
    b = np.array(b+[1])
    print('Verify F', a, b, '->', np.dot(np.dot(a.T, F),b))
print('Corresponding points:')
print(len(points1))
print('Essential matrix:')
print(E)
print('points1')
print(points1)
print('points2')
print(points2)

def step3():
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    U, s, V = np.linalg.svd(E)
    print('Singular values:', s)
    return [(np.dot(np.dot(U, w), V.T), u) for w,u in itertools.product([W, W.T], [U[:, -1], -U[:, -1]])]
    
Ps = step3()
print('Possible projections:')
for R, t in Ps:
    print(R, '&', t)

def step4(P):
    X = list()
    for a, b in zip(points1, points2):
        A = np.array([
            a[0]*np.array([0,0,1,0])-np.array([1,0,0,0]),
            a[1]*np.array([0,0,1,0])-np.array([0,1,0,0]),
            b[0]*P[2,:]-P[0,:],
            b[1]*P[2,:]-P[1,:]
        ])
        point3d = np.linalg.svd(A)[2][:,3]
        point3d /= point3d[-1]
        X.append(point3d[:-1])
    return np.array(X)

print('3D points:')
for j in [3]:
    points3d = step4(np.hstack([Ps[j][0], Ps[j][1].reshape(3,1)]))
    print(points3d)

def render3d():
    theta = 0.01
    t = np.array([80, -180.0, 0.0]).reshape((3,1))
    ct = 0
    while(1):
        R = cv2.Rodrigues(np.array([0.0,0.0,theta]).reshape(3,1))[0]
        P = np.dot(np.diag([0.001, 0.001, 1]), np.hstack([R, t]))
        points2d = [np.dot(np.dot(K, P), np.array(X.tolist()+[1])) for X in points3d]
        if ct==0: print('2D points:')
        canvas = np.zeros((600,800,3), np.uint8)
        canvas[:,:] = (255,255,255)
        for x in points2d:
            _x = np.round(x[:2]).astype(int)
            _x[1] = -_x[1]
            if ct==0: print(_x)
            cv2.circle(canvas, tuple(_x), 2, (128,128,0))
        cv2.imshow('result', canvas)

        key = cv2.waitKey(33) & 0xFF
        if key == ord('\x1B'):
            sys.exit()
        elif key== ord('a'):
            t+=np.array([-1.0,0,0]).reshape((3,1))
            print(t)
        elif key== ord('d'):
            t+=np.array([1.0,0,0]).reshape((3,1))
        elif key== ord('w'):
            t+=np.array([0,1.0,0]).reshape((3,1))
        elif key== ord('s'):
            t+=np.array([0,-1.0,0]).reshape((3,1))
        elif key== ord('q'):
            theta+=0.06
        elif key== ord('e'):
            theta-=0.06
        ct += 1
#render3d()

while(1):
    key = cv2.waitKey(33) & 0xFF
    if key == ord('\x1B'):
        sys.exit()

