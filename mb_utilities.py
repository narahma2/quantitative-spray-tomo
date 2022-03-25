import astra
import cv2
import h5py
import numpy as np

from matplotlib import pyplot as plt
from skimage.measure import (
                             compare_nrmse as nrmse,
                             compare_psnr as psnr,
                             compare_ssim as ssim
                             )
from tomopy import find_center


# Data directories
inp_fld = 'inputs'
out_fld = 'outputs'


def convertMB(raw_file, loc, ind):
    """
    Script to convert the raw time-averaged scans into projected density.

    :param str raw_file: Path to the HDF5 file.
    :param str loc: Which axial location to load (jets/impingement/near).
    :param int ind: Index to start calculating background offset from.
    :returns: Tuple containing angle, x location, and projected density data.
    :rtype: (numpy.array(float), numpy.array(float), numpy.array(float))
    """
    # Load in averaged HDF5 results
    with h5py.File(f'{inp_fld}/{raw_file}.hdf5') as hf:
        # Angular rotation (radians)
        theta = hf[loc]['theta'][()]

        # X locations (mm)
        x = hf[loc]['x'][()]

        # Averaged extinction lengths (ln(I0/I))
        # First dimension is the angle, second is the X location
        ext_len = hf[loc]['ext_length'][()]

    # Calculate background offset (edge of the scan outside the spray)
    offset = np.nanmedian(ext_len[:, ind:])

    # Apply background correction
    ext_len -= offset

    # Attenuation coefficient (total w/o coh. scattering - cm^2/g)
    # Convert to mm^2/g for mm
    # Pure water @ 8 keV
    # <https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html>
    atten_coeff = (1.006*10*(10*10))

    # Calculated projected density in micrograms/mm^2 using Beer-Lambert law
    proj_density = (ext_len / atten_coeff) * 1E6

    # Mask projections
    proj_density[proj_density < 50] = 0

    return theta, x, proj_density


def showSprayMB(loc, mb, view, Re):
    """
    Visualize the projection scan data.

    :param str loc: Which axial location to load (jets/impingement/near).
    :param tuple mb: The low and high flowrate projected density scans.
    :param int view: Which view to load.
    :param int Re: Reynolds number of current condition.
    """
    # Unpack inputs
    theta = np.rad2deg(mb[loc][0][view]).astype(int)
    x = mb[loc][1]
    mbDens = mb[loc][2]

    # Setup figure
    fig = plt.figure()
    plt.plot(x, mbDens[view,:])

    # Set title
    plt.title(f'Projection Scans - $Re$ = {Re} - $\\theta$ = {theta}°')
    plt.xlabel('X [mm]')
    plt.ylabel('Projected density [μg/mm$^2$]')
    plt.show()

    return


def scaleTIM(mb):
    """
    Scale the projected density scans to have consistent liquid masses.

    :param tuple mb: The low and high flowrate projected density scans.
    :returns: Tuple of scaled projected density scan and error before/after.
    :rtype: (numpy.array(float), (float,), (float,))
    """
    proj = mb[2].copy()

    scaleMean = proj.sum(axis=1).mean()
    cfMean = scaleMean / proj.sum(axis=1)
    errMean = 100*(abs(scaleMean - proj.sum(axis=1)) / scaleMean).mean()
    proj *= cfMean[:,np.newaxis]

    scaleMean = proj.sum(axis=1).mean()
    corrMean = 100*(abs(scaleMean - proj.sum(axis=1)) / scaleMean).mean()

    return proj, errMean, corrMean


def rotCenter(mb, c0):
    """
    Find the rotation center of the projection data.

    :param tuple mb: The low and high flowrate projected density scans.
    :param float c0: Initial guess of where center is located.
    :returns: Rotation center.
    :rtype: float
    """
    center = find_center(mb[2][:,np.newaxis,:], mb[0], init=c0, tol=0.1)

    return center


def recon(mb, win, center):
    """
    Iterative reconstruction routine that reconstructs each of the slices.

    :param tuple mb: Angle, x locations, and measured projection data.
    :param tuple win: Windows to use that define the reconstruction grid.
    :param float center: Rotation center.
    :returns: Tuple containing reconstructed slice, reprojection, and history.
    :rtype: (numpy.array(float), numpy.array(float), list(dict))
    """
    theta = mb[0]
    x = mb[1]
    proj = mb[2]

    dx = x[1] - x[0]

    winX = win[0]
    winY = win[1]

    # Setup ASTRA geometry
    proj_geom = astra.create_proj_geom('parallel', dx, x.size, theta)
    proj_geom = astra.geom_postalignment(proj_geom, x.size/2 - center)

    voxelSize = [dx, dx]
    volSize = np.array([
                        np.ceil(np.array(winX).ptp()/voxelSize[0]),
                        np.ceil(np.array(winY).ptp()/voxelSize[1])
                        ], dtype=int)
    vol_geom = astra.create_vol_geom(
                                     volSize[1], volSize[0],
                                     winX[0], winX[1],
                                     winY[0], winY[1]
                                     )
    proj_id = astra.create_projector('strip', proj_geom, vol_geom)
    dA = np.prod(np.array([np.ptp(winX), np.ptp(winY)]) / volSize)
    gridX = np.linspace(winX[0], winX[1], volSize[0])
    gridY = np.linspace(winY[0], winY[1], volSize[1])

    # Get system matrix
    W = astra.optomo.OpTomo(proj_id)

    # MLOS routine
    numViews = np.shape(proj)[0]
    v_views = np.zeros([numViews] + list(W.vshape))

    for i in range(numViews):
        proj_view = np.zeros_like(proj)
        proj_view[i,:] = (proj[i,:] > 0).astype(np.float32)
        v_views[i,:] = W.BP(proj_view)

    v_views = v_views > 0
    MLOS = (np.sum(v_views, axis=0) >= numViews-1).astype(np.float32)

    # MLEM algorithm
    volMask = MLOS.copy()
    v = volMask.copy()
    onesBP = W.BP(np.ones_like(proj))
    weightsV = np.divide(1, onesBP,
                         out=np.zeros_like(onesBP),
                         where=onesBP!=0)

    reproj = W.FP(v)
    iters = 1000
    errors = [None] * iters
    stopCrit = [None] * iters

    for i in range(iters):
        norm1 = np.linalg.norm(reproj, axis=1)
        ratio = np.divide(proj, reproj,
                          out=np.zeros_like(reproj),
                          where=reproj!=0)
        cf = W.BP(ratio) * weightsV
        v *= np.fmax(cf, 0)
        C = get_2D_filter(mu=5, poly=3, order=0)
        v = cv2.filter2D(src=v, ddepth=-1, kernel=C)
        v = v.clip(0, 997)
        v *= volMask.clip(0,1)
        v[~np.isfinite(v)] = 0

        reproj = W.FP(v)
        reproj[np.isnan(reproj)] = 0
        reproj[np.isinf(reproj)] = 0

        norm2 = np.linalg.norm(reproj, axis=1)
        stopCrit[i] = np.mean(np.abs(norm2 - norm1)) / np.mean(norm2)
        errors[i] = get_errors(proj, reproj)

        if stopCrit[i] < 1E-4:
            break

    errors = [x for x in errors if np.any(x)]
    stopCrit = stopCrit[:len(errors)]
    iters = len(errors)

    astra.clear()

    # History statistics tracking progression of reconstruction
    histDict = {
                'Iterations needed': iters,
                'Projection errors': errors,
                'L2-norm differences': stopCrit
                }

    return v, reproj, histDict


def showHistory(hist, loc, Re):
    """
    Plot the history information.

    :param dict hist: History information.
    :param str loc: Which location to load (jets/impingement/near).
    :param int Re: Reynolds number of current condition.
    """
    # Unpack inputs
    iters = hist[loc]['Iterations needed']
    err = hist[loc]['Projection errors']
    crit = hist[loc]['L2-norm differences']

    # Calculate errors/convergence mean/std
    x = np.arange(0, iters)
    nrmse_mean = 100*np.array([x['NRMSE'].mean() for x in err])
    nrmse_std = 100*np.array([x['NRMSE'].std() for x in err])
    conv_mean = 100*np.array([x.mean() for x in crit])
    conv_std = 100*np.array([x.std() for x in crit])

    # Setup figure
    fig, ax = plt.subplots(2, 1, sharex=True)

    # Plot errors
    ax[0].plot(x, nrmse_mean)
    ax[0].fill_between(
                     x,
                     nrmse_mean - nrmse_std, nrmse_mean + nrmse_std,
                     alpha=0.25
                     )

    # Plot convergence
    ax[1].plot(x, conv_mean)
    ax[1].fill_between(
                     x,
                     conv_mean - conv_std, conv_mean + conv_std,
                     alpha=0.25
                     )
    ax[1].set_yscale('log')

    # Labels
    ax[0].set_ylabel('NRMSE (%)')
    ax[1].set_ylabel('$L_2$-norm differences')
    ax[1].set_xlabel('Iteration')

    # Title
    fig.suptitle(f'Solution history - $Re$ = {Re}')
    plt.show()

    return


def showComparisonMB(mb, reproj, loc, view, Re):
    """
    Visualize the projection and reprojection data.

    :param tuple mb: The measured projected density scans.
    :param tuple reproj: Reprojected density scans at different locations.
    :param str loc: Which axial location to load (jets/impingement/near).
    :param int view: Which view to load.
    :param int Re: Reynolds number of current condition.
    """
    # Unpack inputs
    theta = np.rad2deg(mb[loc][0][view]).astype(int)
    x = mb[loc][1]
    proj = mb[loc][2][view]
    reproj = reproj[loc][view]

    # Plot projections/reprojections
    plt.plot(x, proj, label='Projection')
    plt.plot(x, reproj, label='Reprojection')

    # Labels/titles/legend
    plt.xlabel('x [mm]')
    plt.ylabel('Projected density [µg/mm$^2$]')
    plt.title(f'Projection Comparison - $Re$ = {Re} - $\\theta$ = {theta}°')
    plt.legend()
    plt.show()

    return


def showLVF(v, win, loc, Re):
    """
    Show LVF slices.

    :param numpy.array(float) v: Reconstructed slice array at each location.
    :param tuple win: Windows to use that define the reconstruction grid.
    :param str loc: Which axial location to load (jets/impingement/near).
    :param int Re: Reynolds number of current condition.
    """
    # Unpack inputs
    v = v[loc].copy() / 997
    win = win[loc]

    # Set background to white
    v[v == 0] = np.nan

    # Plot slice
    plt.imshow(
               v,
               vmin=0.05,
               vmax=0.95,
               cmap='YlGnBu',
               extent=(win[0] + win[1]),
               aspect='equal'
               )

    # Display colorbar
    plt.colorbar(label='LVF [-]')

    # Labels/title
    plt.xlabel('z [mm]')
    plt.ylabel('x [mm]')
    plt.title(f'Slice - $Re$ = {Re}')
    plt.show()

    return


def get_2D_filter(mu, poly, order=0):
    """
    Loads the 2D Savitzky-Golay filter of size mu with poly-th polynomial
    order.

    :param int mu: Size of the mu*mu kernel.
    :param int poly: Order of the best-fit polynomial to be used.
    :param int order: Order of the derivative to be calculated. order=0
        returns the filter to calculate the filtered value at the central node
        of the kernel.
    :returns: A numpy array of shape (mu, mu) containing the desired filter.
    :rtype: numpy.array(float)
    """
    smu = str(mu).zfill(3)
    spoly = str(poly).zfill(3)

    #Open file
    f = open(f'{inp_fld}/CC_{smu}x{smu}_{spoly}x{spoly}.dat', 'r')

    #Go through file
    while 1:
        s = f.readline()
        if s=='# Matrix starts:\n':
            break

    #Skip lines until you're at the right order.
    for o in np.arange(order):
        f.readline()

    s = f.readline()
    s = s.split('\t')
    f.close()

    #Find out what the symmetry of the line is.
    symmetry = s[1]

    #Populate the first half of the filter from the stored numbers.
    s = np.array(s[2:]).astype(np.float)

    #Populate  thesecond half based on symmetry. These are centred filters, so
    #leave the central column out.
    if symmetry =='S':
        s = np.append(s,s[0:-1][::-1])
    elif symmetry == 'A':
        s = np.append(s,-1.0*s[0:-1][::-1])

    C = s.reshape((mu,mu))

    return C


def get_errors(proj, reproj):
    """
    Calculate errors between the measured projection and reprojections.

    :param numpy.array(float) proj: Measured projection data.
    :param numpy.array(float) reproj: Reconstructed projection data.
    :returns: Dict of error parameters.
    :rtype: dict.
    """
    errNRMSE = np.array([nrmse(x,y) for x,y in zip(proj, reproj)])
    errMass = np.array([nrmse(x.sum(), y.sum()) for x,y in zip(proj, reproj)])
    errPSNR = np.array([
                        psnr(x, y, data_range=y.ptp())
                        for x,y in zip(proj, reproj)
                        ])
    errSSIM = np.array([
                        ssim(x, y, data_range=y.ptp())
                        for x,y in zip(proj, reproj)
                        ])

    return {'NRMSE': errNRMSE, 'Mass': errMass,
            'PSNR': errPSNR, 'SSIM': errSSIM}
