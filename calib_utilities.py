import cv2
import h5py
import numpy as np

import matplotlib.pyplot as plt
from skimage.filters import threshold_local


# Data directories
inp_fld = 'inputs'
out_fld = 'outputs'


def processCamera(cam, gridSize, centerPt, flip, useMask):
    """
    Main function that finds the dots in each of the camera's views.

    :param str cam: Field name to load from HDF5 file.
    :param (int, int) gridSize: Number of circles (rows x columns).
    :param int centerPt: Origin dot index in the grid array.
    :param bool flip: Whether to flip the image (camera 2).
    :param bool useMask: Whether to mask the image.
    :returns: Detected circle positions as (image [px], world [mm]) coords.
    :rtype: (numpy.array(float), numpy.array(float))
    """
    # Load calibration data
    pbCal = getCalibration(cam)

    # Define the object points (dot positions in world coordinates)
    totPts = gridSize[0] * gridSize[1]
    objp = np.zeros((totPts, 3), np.float32)
    objp[:, :2] = np.mgrid[0:gridSize[0], 0:gridSize[1]].T.reshape(-1, 2)

    # Dot-to-dot distance on the calibration plate (mm)
    spacing = 3.7

    # Center the X axis using an origin common to all views and convert to mm
    # Center points were found manually before cropping the images
    # See Original/Camera# under the calibration.hdf5 file -- origin is taken
    # to be underneath the "long" cut-out mark on the top of the plate.
    # Flipping done for Camera 2 (it's looking on the back side of the plate)
    objPts = np.array([
                       [flip*spacing*(x[0]-centerPt), spacing*x[1], x[2]]
                       for x in objp
                       ])

    # Get image points (object points are all the same within each camera)
    imgPts = [detectPoints(x, gridSize, m) for x,m in zip(pbCal, useMask)]

    # Convert to cv2.calibrateCamera() format
    # Vector of vectors: https://stackoverflow.com/a/49172755/16310149
    # Must be a float: https://stackoverflow.com/a/58116723/16310149
    objPts = np.repeat([objPts], repeats=3, axis=0).astype('float32')
    imgPts = np.array(imgPts).astype('float32')

    return imgPts, objPts


def showPoints(cam, view, imgPts, objPts):
    """
    Visualization function to show dot finding results.

    :param int cam: Which camera to load (1, 2, 3).
    :param int view: Which calibration plate view to load (1, 2, 3).
    :param numpy.array(float) imgPts: Detected circle positions [px].
    :param numpy.array(float) objPts: Detected circle positions [mm].
    """
    # Plot calibration image
    img = getCalibration(f'Cropped/Camera{cam}')[view-1]
    plt.imshow(img, cmap='gray')

    # Plot initial points
    plt.plot(
             *zip(*imgPts[cam-1][view-1][:, 0, :]),
             color='r',
             linestyle='',
             marker='o',
             markersize=4,
             alpha=0.5
             )
    plt.title(f'Camera {cam} - View {view}')
    plt.axis('off')
    plt.show()

    return


def calibCamera(imgPts, objPts):
    """
    Main function that runs through calibration routine (Zhang's method).

    :param numpy.array(float) imgPts: Detected circle positions [px].
    :param numpy.array(float) objPts: Detected circle positions [mm].
    :returns: Calibration parameters and errors/standard deviations.
    :rtype: (tuple(dict), tuple(dict))
    """
    # Image size in pixels
    imSize = (512, 512)

    # Run through calibration
    # Outputs are: mean reprojection error, intrinsic matrix, distortion
    # coeff., rotation vectors (radians), translation vectors (mm)
    flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL + \
            cv2.CALIB_TILTED_MODEL
    out = cv2.calibrateCameraExtended(
                                      objPts,
                                      imgPts,
                                      imSize,
                                      None,
                                      None,
                                      flags=flags
                                      )

    # Unpack outputs
    meanErr = out[0]
    mtx = out[1]
    dist = out[2]
    rvecs = out[3]
    tvecs = out[4]
    intStDev = out[5]
    extStDev = out[6]
    viewErr = out[7]

    # Collect calibration information into a dictionary
    camCal = {
              'Intrinsics': mtx,
              'Distortion': dist,
              'Rotation': rvecs,
              'Translation': tvecs
              }

    # Standard deviation values
    errCal = {
              'Intrinsics StDev': intStDev.flatten(),
              'Extrinsics StDev': extStDev.flatten(),
              'View Errors': viewErr.flatten()
              }

    return camCal, errCal


def showCalibration(ind, camCal, errCal):
    """
    Output the calibration parameters and associated uncertainties (3*std).

    :param int ind: Which camera to output (1, 2, 3).
    :param tuple(dict) camCal: Calibration parameters for each view.
    :param tuple(dict) errCal: Calibration errors/deviations for each view.
    """
    cam = camCal[ind-1]
    err = errCal[ind-1]

    # Focal lengths (+/- 3*std. dev.)
    fl = (cam['Intrinsics'][0,0], cam['Intrinsics'][1,1])
    fl_3std = (3*err['Intrinsics StDev'][0], 3*err['Intrinsics StDev'][1])

    # Principal points (+/- 3*std. dev.)
    pp = (cam['Intrinsics'][0,2], cam['Intrinsics'][1,2])
    pp_3std = (3*err['Intrinsics StDev'][2], 3*err['Intrinsics StDev'][3])

    # Radial model distortion coefficients (+/- 3*std. dev.)
    r = cam['Distortion'][0][[0,1,4]]
    r_3std = 3*err['Intrinsics StDev'][[0,1,4]]

    # Tangential model distortion coefficients (+/- 3*std. dev.)
    p = cam['Distortion'][0][[2,3]]
    p_3std = 3*err['Intrinsics StDev'][[2,3]]

    # Rational model distortion coefficients (+/- 3*std. dev.)
    k = camCal[0]['Distortion'][0][[5,6,7]]
    k_3std = 3*errCal[0]['Intrinsics StDev'][[5,6,7]]

    # Thin prism model distortion coefficients (+/- 3*std. dev.)
    tp = cam['Distortion'][0][[8,9,10,11]]
    tp_3std = 3*err['Intrinsics StDev'][[8,9,10,11]]

    # Tilted model distortion coefficients (+/- 3*std. dev.)
    t = cam['Distortion'][0][[12,13]]
    t_3std = 3*err['Intrinsics StDev'][[12,13]]

    # Rotations (+/- 3*std. dev.)
    R = [np.rad2deg(x.flatten()) for x in cam['Rotation']]
    Rind = [0,1,2,6,7,8,12,13,14]
    R_3std = np.rad2deg(err['Extrinsics StDev'][Rind]).reshape((3,3))

    # Translations (+/- 3*std. dev.)
    T = [x.flatten() for x in cam['Translation']]
    Tind = [3,4,5,9,10,11,15,16,17]
    T_3std = err['Extrinsics StDev'][Tind].reshape((3,3))

    print('')
    print(f'Calibration Results')
    print(f'===================')
    print('')

    print('Intrinsics')
    print('----------')

    print(
          'Focal Lengths (x, y) [px]: ({0:0.1f}, {1:0.1f}) +/- '
          '({2:0.1f}, {3:0.1f})'.format(*fl, *fl_3std)
          )

    print(
          'Principal points (x, y) [px]: ({0:0.1f}, {1:0.1f}) +/- '
          '({2:0.1f}, {3:0.1f})'.format(*pp, *pp_3std)
          )
    print('')

    print('Distortion Coefficients')
    print('-----------------------')
    print(
          'Radial [-]: ({0:0.2f}, {1:0.2f}, {2:0.2f}) +/- '
          '({3:0.2f}, {4:0.2f}, {5:0.2f})'.format(*r, *r_3std)
          )

    print(
          'Tangential [-]: ({0:0.3f}, {1:0.3f}) +/- '
          '({2:0.3f}, {3:0.3f})'.format(*p, *p_3std)
          )

    print(
          'Rational [-]: ({0:0.3f}, {1:0.3f}, {2:0.3f}) +/- '
          '({3:0.3f}, {4:0.3f}, {5:0.3f})'.format(*k, *k_3std)
          )

    print(
          'Thin prism [-]: ({0:0.3f}, {1:0.3f}, {2:0.3f}, {3:0.3f}) +/- '
          '({4:0.3f}, {5:0.3f}, {6:0.3f}, {7:0.3f})'.format(*tp, *tp_3std)
          )

    print(
          'Tilted [-]: ({0:0.3f}, {1:0.3f}) +/- '
          '({2:0.3f}, {3:0.3f})'.format(*t, *t_3std)
          )
    print('')

    print('Extrinsics')
    print('----------')
    print(
          'View 1 rotations (x, y, z) [deg]: ({0:0.1f}, {1:0.1f}, {2:0.1f}) '
          '+/- ({3:0.1f}, {4:0.1f}, {5:0.1f})'.format(*R[0], *R_3std[0])
          )
    print(
          'View 1 translations (x, y, z) [mm]: ({0:0.1f}, {1:0.1f}, {2:0.1f})'
          ' +/- ({3:0.1f}, {4:0.1f}, {5:0.1f})'.format(*T[0], *T_3std[0])
          )
    print('')

    print(
          'View 2 rotations (x, y, z) [deg]: ({0:0.1f}, {1:0.1f}, {2:0.1f}) '
          '+/- ({3:0.1f}, {4:0.1f}, {5:0.1f})'.format(*R[1], *R_3std[1])
          )
    print(
          'View 2 translations (x, y, z) [mm]: ({0:0.1f}, {1:0.1f}, {2:0.1f})'
          ' +/- ({3:0.1f}, {4:0.1f}, {5:0.1f})'.format(*T[1], *T_3std[1])
          )
    print('')

    print(
          'View 3 rotations (x, y, z) [deg]: ({0:0.1f}, {1:0.1f}, {2:0.1f}) '
          '+/- ({3:0.1f}, {4:0.1f}, {5:0.1f})'.format(*R[2], *R_3std[2])
          )
    print(
          'View 3 translations (x, y, z) [mm]: ({0:0.1f}, {1:0.1f}, {2:0.1f})'
          ' +/- ({3:0.1f}, {4:0.1f}, {5:0.1f})'.format(*T[2], *T_3std[2])
          )

    return


def showReprojection(cam, view, imgPts, objPts, camCal, errCal):
    """
    Visualization function to show reprojections.

    :param int cam: Which camera to load (1, 2, 3).
    :param int view: Which calibration plate view to load (1, 2, 3).
    :param numpy.array(float) imgPts: Detected circle positions [px].
    :param numpy.array(float) objPts: Detected circle positions [mm].
    :param tuple(dict) camCal: Calibration parameters for each view.
    :param tuple(dict) errCal: Calibration errors/deviations for each view.
    """
    # Calculate reprojected points from calibration parameters
    reproj, _ = cv2.projectPoints(
                                  objPts[cam-1][view-1],
                                  camCal[cam-1]['Rotation'][view-1],
                                  camCal[cam-1]['Translation'][view-1],
                                  camCal[cam-1]['Intrinsics'],
                                  camCal[cam-1]['Distortion']
                                  )

    # Plot calibration image
    img = getCalibration(f'Cropped/Camera{cam}')[view-1]
    plt.imshow(img, cmap='gray')

    # Plot initial points
    plt.plot(
             *zip(*imgPts[cam-1][view-1][:, 0, :]),
             color='r',
             linestyle='',
             marker='o',
             markersize=8,
             alpha=0.5,
             label='Initial'
             )

    # Plot reprojected points
    plt.plot(
             *zip(*reproj[:, 0, :]),
             color='b',
             linestyle='',
             marker='x',
             markersize=8,
             label='Reprojection'
             )
    plt.title(f'Camera {cam} - View {view}: Mean Pixel Error = '
              f'{errCal[cam-1]["View Errors"][view-1]:0.3f}')
    plt.legend()
    plt.axis('off')
    plt.show()

    return


def getCalibration(cam):
    """
    Retrieve calibration images. The images were manually masked in ImageJ
    beforehand so that all views for each camera looked at the same dots as
    this was just easier to setup. The original un-masked images are also
    available in the HDF5 file.

    :param str cam: Field name to load from HDF5 file.
    :returns: Array containing camera's calibration views.
    :rtype: numpy.array(int)
    """
    # Path to calibration data
    pbFile_cal = f'{inp_fld}/calibration.hdf5'

    with h5py.File(pbFile_cal, 'r') as hf:
        data = hf[cam][()]

    return data


def makeBlob():
    """
    Create OpenCV blob that is used to detect circles. Values were
    pre-determined, play around with settings if different images are used!

    :returns: OpenCV blob detector object for circle detection.
    :rtype: cv2.SimpleBlobDetector_Params
    """
    # Setup SimpleBlobDetector parameters
    # Look up OpenCV documentation for additional parameters
    blobParams = cv2.SimpleBlobDetector_Params()

    # Filter by Area
    blobParams.filterByArea = True
    blobParams.minArea = 50//4
    blobParams.maxArea = 1200//4

    # Filter by Convexity
    blobParams.filterByConvexity = True
    blobParams.minConvexity = 0.10

    # Filter by Inertia
    blobParams.filterByInertia = True
    blobParams.minInertiaRatio = 0.01

    # Filter by color
    blobParams.filterByColor = True
    blobParams.blobColor = 0

    # Create a detector with the parameters
    blobDetector = cv2.SimpleBlobDetector_create(blobParams)

    return blobDetector


def detectPoints(img, gridSize, useMask):
    """
    Detect circles on each of the calibration plate views for desired camera.

    :param numpy.array(int) img: Single calibration plate view.
    :param (int, int) gridSize: Number of circles (rows x columns).
    :param bool useMask: Whether to mask the image.
    :returns: Detected circle positions in image coordinates [px].
    :rtype: numpy.array(float)
    """
    # Mask out View 3 for Cameras 1 and 3 only
    if useMask:
        mask = (img == 255).astype(img.dtype)
        img = ~mask * (img > threshold_local(img, 21, method='mean'))
        img = img.astype(np.uint8)

    # Convert to gray scale with dark blobs
    gray = cv2.convertScaleAbs(img)
    gray = cv2.convertScaleAbs(255 - gray)

    # Get the blob
    blobDetector = makeBlob()

    # Detect blobs
    keypoints = blobDetector.detect(gray)

    # Draw detected blobs as red circles
    # This helps cv2.findCirclesGrid()
    flag = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    im_with_keypoints = cv2.drawKeypoints(
                                          gray,
                                          keypoints,
                                          np.array([]),
                                          (255, 0, 0),
                                          flag
                                          )
    im_with_keypoints_gray = cv2.cvtColor(
                                          im_with_keypoints,
                                          cv2.COLOR_BGR2GRAY
                                          )

    # Keypoint array
    kypt = np.array([x.pt for x in keypoints])

    # Find the circle grid
    _, corners = cv2.findCirclesGrid(
                                     im_with_keypoints_gray,
                                     gridSize,
                                     kypt,
                                     flags=cv2.CALIB_CB_SYMMETRIC_GRID
                                     )

    return corners
