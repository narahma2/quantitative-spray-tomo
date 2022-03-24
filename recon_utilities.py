import astra
import copy
import h5py
import numpy as np
import pyvista as pv
import warnings

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyevtk.hl import imageToVTK
from scipy.ndimage import (
                           gaussian_filter,
                           median_filter
                           )
from skimage.measure import (
                             label,
                             regionprops,
                             compare_nrmse,
                             compare_psnr,
                             compare_ssim
                             )


# Data directories
inp_fld = 'inputs'
out_fld = 'outputs'


def loadSpray(path):
    """
    Load in the pre-processed projected density images.

    :param str path: File path to the HDF5 projected density arrays.
    :returns: Tuple containing the low and high Reynolds number projections.
    :rtype: (numpy.array(float), numpy.array(float))
    """
    with h5py.File(path, 'r') as hf:
        pbDens_lo = hf['Re-7100'][()]
        pbDens_hi = hf['Re-10600'][()]

    return pbDens_lo, pbDens_hi


def getSystemMatrix(geom):
    """
    Retrieve system matrix (W) that defines the mapping process between volume
    (x) and projections (p): Wx = p. The matrix is built on a cone beam
    projector using the ASTRA toolbox.
    References:
        [1] van Aarle, W. et al. The ASTRA Toolbox: A platform for advanced
            algorithm development in electron tomography. Ultramicroscopy 157,
            35–47 (2015). doi: 10.1016/j.ultramic.2015.05.002.
        [2] van Aarle, W. et al. Fast and flexible X-ray tomography using the
            ASTRA toolbox. Opt. Express 24, 25129 (2016).
            doi: 10.1364/OE.24.025129.

    :param list(dict): Geometry dict for each line of sight.
    :returns: System matrix from ASTRA toolbox.
    :rtype: astra.optomo.OpTomo
    """
    # Check if CUDA is available
    if not astra.astra.use_cuda():
        # CUDA is not available, return None
        print('CUDA not detected! Execute astra.test() in Python to see'
              ' current functionality--is there a GPU present and compatible'
              ' with the current version of ASTRA?')

        return None

    # Initialize vectors
    vectors = np.zeros((len(geom), 12), dtype=np.float32)

    for i, _ in enumerate(vectors):
        vectors[i, [0, 1, 2]] = geom[i]['Anode']
        vectors[i, [3, 4, 5]] = geom[i]['Detector']
        vectors[i, [6, 7, 8]] = geom[i]['U Axis']
        vectors[i, [9, 10, 11]] = geom[i]['V Axis']

    # Range of X, Y, Z in mm over the entire volume
    # X and Y are taken from the radiograph images; the Z window was refined
    # by inspecting the resulting reconstruction and tuning the values
    volX = np.array([-10, 10])
    volY = np.array([-9, 20])
    volZ = np.array([-15, 15])

    # Chosen grid size (similar to the detector size)
    voxelSize = [0.1, 0.1, 0.1]

    # Set reconstruction grid size
    # Length of the OVERALL volume grid
    volSize = np.array([
                        volX.ptp()/voxelSize[0],
                        volY.ptp()/voxelSize[1],
                        volZ.ptp()/voxelSize[2]
                        ], dtype=int)
    gridX = np.linspace(volX[0], volX[1], volSize[0])
    gridY = np.linspace(volY[0], volY[1], volSize[1])
    gridZ = np.linspace(volZ[0], volZ[1], volSize[2])

    # Setup ASTRA geometry
    proj_geom = astra.create_proj_geom('cone_vec', 512, 512, vectors)
    vol_geom = astra.create_vol_geom(volSize[1], volSize[0], volSize[2],
                                     volX[0], volX[1],
                                     volY[0], volY[1],
                                     volZ[0], volZ[1])

    # 3D projectors: cuda3d
    # blob not implemented, linear3d and linearcone not available(?)
    proj_id = astra.create_projector('cuda3d', proj_geom, vol_geom)

    # Get the OpTomo representation of the system matrix for brevity
    W = astra.optomo.OpTomo(proj_id)

    # Clearing astra variables (only need W)
    astra.clear()

    return W


def MLOS(W, proj, constraint, numLOS, fp=None):
    """
    Run through line-of-sight initialization routine.
    References:
        [1] Atkinson, C. & Soria, J. An efficient simultaneous reconstruction
            technique for tomographic particle image velocimetry.
            Exp. Fluids 47, 553–568 (2009). doi: 10.1007/s00348-009-0728-0.
        [2] Worth, N. A. & Nickels, T. B. Acceleration of Tomo-PIV by
            estimating the initial volume intensity distribution.
            Exp. Fluids 45, 847–856 (2008). doi: 10.1007/s00348-008-0504-6.

    :param astra.optomo.OpTomo W: System matrix from ASTRA toolbox.
    :param numpy.ndarray(numpy.uint16) proj: Projection data.
    :param str constraint: Type of constraint to use ('proj' or 'ones').
    :param int numLOS: Number of lines of sight to use as the constraint.
    :param str fp: File path to load previous reconstruction results.
    :returns: Reconstructed volume and re-projection.
    :rtype: (numpy.array(float), numpy.array(float))
    """
    # Check if CUDA is available
    if W is None:
        # CUDA is not available, load in previous results instead
        print('CUDA not detected! Loading previous results.')
        path = f'{inp_fld}/{fp}.npz'
        recon_data = np.load(path)
        v = recon_data['v']
        reproj = recon_data['reproj']

        return v, reproj

    # Volume shape
    volShape = W.vshape

    # Define type of projections to use
    # 'Ones' projection checks all the pixels on the camera
    if constraint == 'ones':
        proj_binned = np.ones_like(proj)
    # 'Proj' projection checks only the populated pixels on the camera
    # This technically changes for every time step in temporally resolved
    # measurements. Consider averaging all the images before using this
    # for a more inclusive answer.
    elif constraint == 'proj':
        proj_binned = (proj > 0).astype(int)

    # Number of projections
    numViews = W.sshape[1]

    # Initialize MLOS volume
    v_views = np.zeros([numViews] + list(volShape))

    for i in range(numViews):
        proj_view = np.zeros_like(proj_binned)
        proj_view[:,i,:] = proj_binned[:,i,:]
        v_views[i,:] = W.BP(proj_view)

    v_views = v_views > 0
    v = np.sum(v_views, axis=0)

    # Apply line-of-sight constraint
    v = (v >= numLOS).astype(np.float32)

    return v, W.FP(v)


def scalarCF(initVol, proj, W, fp):
    """
    Apply scalar correction factor to the projection data based on the jets.

    :param numpy.array(float) initVol: Initial volume data.
    :param numpy.array(float) proj: Projection data.
    :param astra.optomo.OpTomo W: System matrix from ASTRA toolbox.
    :param str fp: File path to save/load reconstruction results.
    :returns: Projection data corrected to the known jet path length.
    :rtype: numpy.array(numpy.float)
    """
    # Check if CUDA is available
    if W is None:
        # CUDA is not available, load in previous results instead
        print('CUDA not detected! Loading previous results.')
        path = f'{inp_fld}/{fp}.npy'
        proj = np.load(path)

        return proj

    # Isolate the jets
    volJets = copy.deepcopy(initVol)
    volJets[:, 63:, :] = 0
    jets = copy.deepcopy(proj) * (W.FP(volJets) > 0)

    # Calculate peak from measured jet projected density values
    jets_peak = [
                 np.apply_along_axis(
                                     lambda v: np.median(v[np.nonzero(v)]),
                                     0,
                                     x.max(axis=1)
                                     ).tolist()
                 for x in jets.swapaxes(0, 1)
                 ]

    # Known jet path length ~ 2000 um (multiply by rho for projected density)
    rho = 1.314
    jets_cf = (rho*2000) / np.array(jets_peak)

    # Apply jet correction
    proj *= jets_cf[np.newaxis, :, np.newaxis]

    return proj


def threshVol(v, mask, bounds):
    """
    Threshold the volume intensities to remove lower intensity noise and
    physically meaningless higher intensities (LVF can't be greater than 1!).

    :param numpy.array(float) v: Volume data.
    :param numpy.array(float) mask: Volume mask data.
    :param (float, float, float) bounds: Lower and upper bounds for each
        iteration and final global lower bound for thresholding.
    :returns: Thresholded volume data.
    :rtype: numpy.array(float)
    """
    # Remove invalid values
    v[np.isnan(v)] = 0
    v[np.isinf(v)] = 0

    # Mask the volume
    v *= mask

    # Threshold values based on location
    # More aggressive for the isolated jets
    imp_loc = 70

    for i in range(imp_loc):
        tmp = v[:,i,:] > 0.3*bounds[1]

        # Isolate regions (if present)
        tmp_labels = label(tmp)
        tmp_regions = regionprops(tmp_labels)

        # There should only be two circular jet regions
        # Crop out smaller regions if any exist
        if len(tmp_regions) > 2:
            # Sort by area (descending)
            tmp_regions.sort(key=lambda x: x.area, reverse=True)

            for rg in tmp_regions[2:]:
                tmp_labels[rg.coords[:,0], rg.coords[:,1]] = 0

            tmp_labels[tmp_labels != 0] = 1
            v[:,i,:] *= tmp_labels

    v[v < bounds[0]] = 0
    v[v > bounds[1]] = bounds[1]

    return v


def get_errors(W, v, proj):
    """
    Calculate the errors between projections and reprojections. Tabulated as
    NRMSE, mass difference, PSNR, and SSIM.

    :param astra.optomo.OpTomo W: System matrix from ASTRA.
    :param numpy.array(float) v: Volume data.
    :param numpy.array(float) proj: Projection data.
    :returns: Dictionary of calculated error values.
    :rtype: dict(float, float, float, float)
    """
    projSize = proj.shape[0] * proj.shape[-1]
    reproj = W.FP(v)

    # Swap axes for easier computation
    proj = np.swapaxes(proj, 0, 1)
    reproj = np.swapaxes(reproj, 0, 1)

    # Calculate error in the entire domain
    spray_range = np.max([
                          proj.max(axis=(1,2)),
                          reproj.max(axis=(1,2))
                          ],
                         axis=0
                         )
    NRMSE = np.array([
                      compare_nrmse(x, y)
                      for (x, y) in zip(proj,reproj)
                      ])
    mass = np.array([
                     compare_nrmse(x.sum(), y.sum())
                     for (x, y) in zip(proj,reproj)
                     ])
    PSNR = np.array([
                     compare_psnr(x, y, data_range=sr)
                     for (x, y, sr) in zip(proj, reproj, spray_range)
                     ])
    SSIM = np.array([
                     compare_ssim(x, y, data_range=sr)
                     for (x, y, sr) in zip(proj, reproj, spray_range)
                     ])

    return {'NRMSE': NRMSE, 'Mass': mass,
            'PSNR': PSNR, 'SSIM': SSIM}


def get_stopCrit(norm1, norm2):
    """
    Calculate difference in L2-norm that is used for convergence.

    :param numpy.array(float) norm1: Pre-iteration L2-norm for each view.
    :param numpy.array(float) norm2: Post-iteration L2-norm for each view.
    :returns: L2-norm difference.
    :rtype: numpy.array(float)
    """
    stopCrit = np.divide(
                         np.abs(norm2 - norm1),
                         norm2,
                         out=np.zeros_like(norm1),
                         where=norm2!=0
                         )

    return stopCrit


def recon(W, v, proj, mask, bounds, iters, stop, fp):
    """
    Intermediate function that checks whether or not CUDA is present before
    calling the main reconstruction routine. If CUDA is not present then
    pre-computed results are loaded. I would recommend to use recon_iter()
    if you want to implement this reconstruction routine into your own
    workflow, this is only for the Jupyter notebook.

    :param astra.optomo.OpTomo W: System matrix from ASTRA.
    :param numpy.array(float) v: Volume data.
    :param numpy.array(float) proj: Projection data.
    :param numpy.array(float) mask: Volume mask data.
    :param (float, float, float) bounds: Lower and upper bounds for each
        iteration and final global lower bound for thresholding.
    :param int iters: Maximum number of iterations.
    :param float stop: Stopping criteria for early termination.
    :param str fp: File path to save/load reconstruction results.
    :returns: Tuple containing volume, reprojection, and solution history
    :rtype: (numpy.array(float), numpy.array(float), dict)
    """
    # Check if CUDA is available
    if W is not None:
        # CUDA is available, run through iterations
        vol, reproj, hist = recon_iter(W, v, proj, mask, bounds, iters, stop)

        # Save NumPy results
        path = f'{out_fld}/{fp}'
        np.savez_compressed(path, vol=vol, reproj=reproj, hist=hist)
    else:
        # CUDA is not available, load in previous results instead
        print('CUDA not detected! Loading previous results.')
        path = f'{inp_fld}/{fp}.npz'
        recon_data = np.load(path, allow_pickle=True)

        # Unpack results
        vol = recon_data['vol']
        reproj = recon_data['reproj']
        hist = recon_data['hist'].tolist()

    return vol, reproj, hist


def recon_iter(W, v, proj, mask, bounds, iters, stop):
    """
    Main function that goes through reconstruction routine. Recommended to
    use this function if implementing this into your own workflow.

    :param astra.optomo.OpTomo W: System matrix from ASTRA.
    :param numpy.array(float) v: Volume data.
    :param numpy.array(float) proj: Projection data.
    :param numpy.array(float) mask: Volume mask data.
    :param (float, float, float) bounds: Lower and upper bounds for each
        iteration and final global lower bound for thresholding.
    :param int iters: Maximum number of iterations.
    :param float stop: Stopping criteria for early termination.
    :returns: Tuple containing volume, reprojection, and solution history
    :rtype: (numpy.array(float), numpy.array(float), dict)
    """
    # Number of views
    numViews = proj.shape[1]

    # Initialize the error/convergence vectors
    errors = [None] * iters
    stopCrit = [None] * iters

    # Weights for the volume correction
    onesBP = W.BP(np.ones_like(proj))
    weightsV = np.divide(np.ones_like(onesBP), onesBP,
                         out=np.zeros_like(onesBP),
                         where=onesBP!=0)

    for i in range(iters):
        # Projections of the reconstructed volume
        reproj = W.FP(v)

        # Calculate pre-update L2-norm for each view
        norm1 = np.linalg.norm(reproj, ord=2, axis=(0,2))

        # Remove invalid values
        reproj[~np.isfinite(reproj)] = 0

        # Calculate ratio between reprojection and projection data
        ratio = np.divide(proj, reproj,
                          out=np.zeros_like(proj),
                          where=reproj!=0)

        # Backproject the ratio and apply volume weights
        cf = W.BP(ratio) * weightsV

        # Update volume (MLEM)
        v *= cf

        # Run through masking/thresholding routine
        v = threshVol(v, mask, bounds)

        # Smooth volume
        v = gaussian_filter(v, sigma=1.2)

        # Get the reprojection errors
        errors[i] = get_errors(W, v, proj)

        # Calculate post-update L2-norm for each view
        norm2 = np.linalg.norm(W.FP(v), ord=2, axis=(0,2))

        # Calculate difference in L2-norm between iterations
        stopCrit[i] = get_stopCrit(norm1, norm2)

        # Check for solution convergence based on stopping criteria
        # Strict requirement of all views used here
        if all(stopCrit[i] < stop):
            break

    # Filter None values (in case of early termination)
    errors = [x for x in errors if np.any(x)]
    stopCrit = stopCrit[:len(errors)]
    iters = len(errors)

    # Final thresholding
    v = threshVol(v, mask, (bounds[2], bounds[1]))

    # Final error
    errors.append(get_errors(W, v, proj))

    # Final L2-norm
    norm2 = np.linalg.norm(W.FP(v), ord=2, axis=(0,2))

    # Final change in L2-norm
    stopCrit.append(get_stopCrit(norm1, norm2))

    # Final number of iterations
    iters += 1

    # History statistics tracking progression of reconstruction
    histDict = {
                'Iterations needed': iters,
                'Projection errors': errors,
                'L2-norm differences': stopCrit
                }

    return v, W.FP(v), histDict


def spatialCF(proj, reproj):
    """
    Calculate spatial correction factor using reconstructions of the
    time-averaged projections.

    :param numpy.array(float) proj: Measured time-averaged projection data.
    :param numpy.array(float) reproj: Reconstructed projection data.
    :returns: Correction factors for each row in the projection image.
    :rtype: numpy.array(numpy.float)
    """
    proj_cf = np.divide(reproj, proj, out=np.zeros_like(proj), where=proj!=0)
    proj_cf = np.array([median_filter(x, 7) for x in proj_cf.swapaxes(0,1)])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        proj_cf[proj_cf == 0] = np.nan
        cf = np.nanmean(proj_cf, axis=2)
        cf[np.isnan(cf)] = 0

    return cf[np.newaxis, :, :].T


def massCheck(proj, geom, label):
    """
    Calculate the total liquid mass and the variation between views.

    :param numpy.array(float) proj: Projection data.
    :param list(dict): Geometry dict for each line of sight.
    :param str label: Label defining the current projection (res/avg, Re).
    :returns: Tuple containing liquid mass per view and coeff. of variation.
    :rtype: (numpy.array(float), float)
    """
    # Liquid masses (micrograms) for each view
    viewMass = [
                x.sum()*geom[i]['Detector Size']**2
                for i,x in enumerate(proj.swapaxes(0,1))
                ]

    # Coefficient of variation in the liquid masses between views
    cv = np.array(viewMass).std() / np.array(viewMass).mean()

    # Display results
    print(f'Measured liquid masses for each view ({label}): '
          f'{np.round(viewMass)} μg')
    print(f'Variation in liquid mass between views ({label}): '
          f'{100*cv:0.2f}%')
    print('')

    return viewMass, cv


def showHistory(state, hist, Re):
    """
    Plot the solution history (NRMSE and convergence).

    :param int state: Which of the spray conditions to show.
    :param (dict, dict) hist: History dict for each of the conditions.
    :param (int, int) Re: Reynolds numbers for each condition.
    :returns: Matplotlib plot.
    """
    # Unpack inputs
    hist = hist[state]
    Re = Re[state]

    # Retrieve statistics
    nrmse = np.array([x['NRMSE'] for x in hist['Projection errors']])
    convergence = np.array([x for x in hist['L2-norm differences']])

    # Setup figure
    fig, ax = plt.subplots(2, 1, sharex=True)
    plt.subplots_adjust(top=0.93, hspace=0.1)

    # Plot errors (NRMSE)
    ax[0].plot(100*nrmse[:,0], label='Camera 1')
    ax[0].plot(100*nrmse[:,1], label='Camera 2')
    ax[0].plot(100*nrmse[:,2], label='Camera 3')
    ax[0].set_ylabel('NRMSE (%)')
    ax[0].legend()

    # Plot convergence
    ax[1].plot(convergence[:,0], label='Camera 1')
    ax[1].plot(convergence[:,1], label='Camera 2')
    ax[1].plot(convergence[:,2], label='Camera 3')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('$L_2$-norm differences')

    fig.suptitle(f'Solution history - $Re$ = {Re}')
    plt.show()

    return


def compareProj(state, proj, reproj, Re, view, cmap):
    """
    Compare the measured projection and reconstructed reprojections.

    :param int state: Which of the spray conditions to show.
    :param numpy.array(float) proj: Projection data.
    :param numpy.array(float) reproj: Reconstructed projection data.
    :param (int, int) Re: Reynolds numbers for each condition.
    :param int view: Which of the cameras to show.
    :param str cmap: Colormap to use for visualizing (from matplotlib).
    :returns: Matplotlib plot.
    """
    # Unpack inputs
    proj = proj[state][:,view]
    reproj = reproj[state][:,view]
    Re = Re[state]

    fig = plt.figure(figsize=(6,3))
    fig.subplots_adjust(top=0.79)
    fig.suptitle(f'Camera {view+1} - $Re$ = {Re}')

    ax = ImageGrid(
                   fig,
                   111,
                   nrows_ncols=(1,1),
                   axes_pad=0.5,
                   cbar_location='right',
                   cbar_mode='each',
                   cbar_size=0.15,
                   cbar_pad=0.1,
                   cbar_set_cax=True
                   )

    # Concatenate the data side-by-side
    im = ax[0].imshow(np.concatenate((proj, reproj), axis=1), cmap=cmap)

    # Set title
    ax[0].set_title('Projection vs. Reprojection [μg/mm$^2$]')

    # Show colorbar
    ax.cbar_axes[0].colorbar(im)

    # Turn off axes
    ax[0].set_axis_off()

    return


def showVolume(state, vol, Re):
    """
    Show select LVF isosurfaces from the reconstructed volume.

    :param int state: Which of the spray conditions to show.
    :param numpy.array(float) vol: Volume data.
    :param (int, int) Re: Reynolds numbers for each condition.
    :returns: PyVista plot.
    """
    # Unpack inputs
    vol = vol[state]
    Re = Re[state]

    # Ensure notebook plotting and not external
    pv.global_theme.notebook = True

    # Load volume into PyVista and visualize
    data = pv.UniformGrid(f'{out_fld}/{vol}.vti')

    # Visualize the 10%, 50%, and 90% LVF isosurfaces
    contours = [0.1, 0.5, 0.9]

    # Apply contours to the dataset, divide by 1000 as the stored values were
    # scaled for storing as 16-bit integers as opposed to floats
    data = data.contour(
                        scalars=data.point_data['LVF']/1000,
                        isosurfaces=contours
                        )

    # Initial camera position
    cpos = [(35.6, -16.1, 39.9), (-0.11, 4.45, 0.02), (-0.25, -0.93, -0.26)]

    # Visualize volume
    pl = pv.Plotter(window_size=[512, 384]) # lower resolution for speed
    pl.camera_position = cpos
    pl.add_mesh(
                data,
                opacity=0.25,
                clim=[0.05, 1],
                cmap='RdBu_r',
                show_scalar_bar=True,
                scalar_bar_args={'title': 'LVF'}
                )
    pl.add_camera_orientation_widget()
    pl.add_text(f'Re = {Re}', position='upper_left')
    pl.show()

    return


def convertVTK(vol, W, name):
    """
    Convert the reconstructed volume into a VTI file for ParaView/PyVista.

    :param numpy.array(float) vol: Volume data.
    :param astra.optomo.OpTomo W: System matrix from ASTRA.
    :param str name: File name for current volume.
    """
    # Check if CUDA is available
    if W is None:
        # CUDA is not available, visualize previous results instead
        print('CUDA not detected! Visualizing previous results.')

        return

    # Retrieve X/Y/Z grid points
    grid = W.vshape[::-1]
    winX = [W.vg['options']['WindowMinX'], W.vg['options']['WindowMaxX']]
    winY = [W.vg['options']['WindowMinY'], W.vg['options']['WindowMaxY']]
    winZ = [W.vg['options']['WindowMinZ'], W.vg['options']['WindowMaxZ']]
    win = [winX, winY, winZ]

    grid = [np.linspace(*w, g) for (w, g) in zip(win, grid)]
    limits = [[-999, 999]] * 3

    # Center the spray such that the impingement point is at (0, 0, 0)
    center_x = -0.37
    center_y = 0.9
    center_z = 0.2

    # Calculate cropping windows
    cropX = [np.argmin(np.abs(grid[0] - x)) for x in limits[0]]
    cropY = [np.argmin(np.abs(grid[1] - y)) for y in limits[1]]
    cropZ = [np.argmin(np.abs(grid[2] - z)) for z in limits[2]]

    # Crop the volume
    vol2 = np.array(
                    vol[
                        cropZ[0]:cropZ[1],
                        cropY[0]:cropY[1],
                        cropX[0]:cropX[1]
                        ],
                    order='F'
                    )

    # Update grid points
    gridX = grid[0][cropX[0]:cropX[1]]
    gridY = grid[1][cropY[0]:cropY[1]]
    gridZ = grid[2][cropZ[0]:cropZ[1]]

    # Calculate voxel size in mm^3
    voxelSize = [
                 np.mean(np.diff(gridX)),
                 np.mean(np.diff(gridY)),
                 np.mean(np.diff(gridZ))
                 ]

    # Location of origin point
    originX = (gridX - center_x)[0]
    originY = (gridY - center_y)[0]
    originZ = (gridZ - center_z)[0]

    # Liquid density
    rho = 1.314

    # Save volume as VTI file
    # Scaled to 1000*liquid volume fraction (LVF) such that LVF of 1 is 1000,
    # LVF of 0.1 is 100, etc.
    imageToVTK(
               f'{out_fld}/{name}',
               origin=[originZ, originY, originX],
               spacing=np.flip(voxelSize, axis=0).tolist(),
               pointData={'LVF': (vol2/rho).astype(np.int16)}
               )

    return
