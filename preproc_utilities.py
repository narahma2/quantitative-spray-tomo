import cv2
import h5py
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.ndimage import (
                           binary_dilation,
                           binary_erosion,
                           median_filter
                           )
from scipy.ndimage.morphology import binary_fill_holes
from skimage.filters import threshold_otsu


# Data directories
inp_fld = 'inputs'
out_fld = 'outputs'


def getDataPB(path, root):
    """
    Load in spray data from HDF5 files.

    :param str path: Path to the HDF5 file.
    :param str root: Attribute to load.
    :returns: Array containing requested data.
    :rtype: numpy.ndarray(numpy.uint16)
    """
    # Load in each camera and stack into a numpy array
    with h5py.File(path, 'r') as hf:
        data = np.stack([hf[root][x][()] for x in hf[root].keys()])

    return data


def convertPB(raw_file, Re, otsu_scale, camCal):
    """
    Main spray processing routine. Images are normalized to transmission
    values, masked, fit against an experimentally determined calibration curve
    to calculate equivalent path length (EPL), and finally converted to
    projected density.

    :param str raw_file: Path to the HDF5 file for the raw spray radiograph.
    :param str Re: Attribute defining corresponding Reynolds number.
    :param list(float) otsu_scale: Scaling factors for the Otsu thresholding.
    :param list(dict) camCal: Camera calibration parameters.
    """
    # Retrieve spray images
    raw = getDataPB(raw_file, '/')

    # Background data for PB images
    # Keys: 'dark-current', 'flat', 'mask'
    pbFile_bg = f'{inp_fld}/spray_polychromatic_background.hdf5'

    # Retrieve dark current images
    dark = getDataPB(pbFile_bg, '/dark-current')

    # Retrieve flat field images
    flat = getDataPB(pbFile_bg, '/flat')

    # Retrieve mask images
    # Masks were manually made in ImageJ to remove scintillator/prism edges
    mask = getDataPB(pbFile_bg, f'/mask/{Re}')

    # Subtract dark field and normalize
    norm = np.divide(raw - dark,
                     flat,
                     out=np.zeros(raw.shape, np.float32),
                     where=flat!=0)

    # Mask image
    norm *= mask

    # Apply median filter to remove salt-and-pepper noise
    # norm is re-calculated later, this one is just to find the spray mask
    filtSize = [8, 5, 9]
    norm = np.array([
                     median_filter(x, filtSize[i])
                     for i,x in enumerate(norm)
                     ])

    # Convert to attenuation levels for further masking
    atten = 1 - norm
    atten[atten < 0] = 0

    # Using Otsu thresholding, so need to convert to integer values
    float2int = 10000
    int2float = 1/float2int
    atten16 = np.rint(atten*float2int).astype(np.uint16)

    # Mask out array for thresholding
    thresh = atten16 * mask

    # Calculate Otsu threshold for each camera
    thresh_otsu = [threshold_otsu(x[np.nonzero(x)]) for x in thresh]

    # Mask out background and retain only the spray foreground
    spray_mask = [
                  x > (otsu_scale[i]*thresh_otsu[i])
                  for i,x in enumerate(thresh)
                  ]

    # Fill any holes to close out the mask
    spray_mask = [binary_fill_holes(x) for x in spray_mask]

    # Dilate the mask to ensure boundary edges of spray are captured
    spray_mask = [binary_dilation(x, iterations=3) for x in spray_mask]

    # Run through erosion/dilation iterations to remove stray pixels/lines
    spray_mask = [binary_erosion(x, iterations=6) for x in spray_mask]
    spray_mask = [binary_dilation(x, iterations=6) for x in spray_mask]

    # Convert mask to NumPy array
    spray_mask = np.array(spray_mask)

    # Apply gain-induced deficit (GID) correction and re-calculate norm
    # GID pre-determined for each camera based on intensifier settings
    gid = np.array([0.01, 0.07, 0.02])
    raw = raw / 65535.0
    raw += (gid[:, np.newaxis, np.newaxis] * raw**2)
    raw *= 65535.0
    norm = np.divide(raw - dark,
                     flat,
                     out=np.zeros(raw.shape, np.float32),
                     where=flat!=0)

    # Apply median filter to remove salt-and-pepper noise
    norm = np.array([
                     median_filter(x, filtSize[i])
                     for i,x in enumerate(norm)
                     ])

    # Interior & exterior mask
    interior = mask.astype(bool)
    exterior = 1 - interior

    # Isolate the background in the masked region of data_norm
    bg = norm * (interior * ~spray_mask)

    # Calculate offset for the normalized (transmission) image
    # Background level should be 1 (X-rays transmission through air)
    # Remember to do it on the nonzero elements only!
    norm_bg_med = np.array([np.median(x[np.nonzero(x)]) for x in bg])

    # Correct for the background offset
    norm /= norm_bg_med[:, np.newaxis, np.newaxis]

    # Coefficients to convert transmission to EPL function
    p = [9.028E+02, -4.355E+03, 8.971E+03, -1.032E+04, 7.304E+03,
         -3.302E+03, 9.690E+02, -1.886E+02, 2.814E+01, 5.484E+00]

    # 9th order polynomial fit to convert transmission to path length
    epl = (
           -np.log(norm, out=np.zeros_like(norm), where=norm!=0)
           /
           np.sum([p[i]*(norm**(9-i)) for i in range(0,10)], axis=0)
           )

    # Convert EPL from centimeter to micron
    epl *= 10000

    # Correct the background to 0 (no liquid)
    bg_epl = epl * ~spray_mask * interior
    offset = np.array([np.median(x[x.nonzero()]) for x in bg_epl])
    epl -= offset[:, np.newaxis, np.newaxis]

    # Remove negative values
    epl[epl < 0] = 0

    # Liquid density of 50% KI mixed with water by mass
    rho = 1.314

    # Convert EPL in microns to projected density in micrograms/mm^2
    proj_density = epl * rho

    # Apply mask
    proj_density *= spray_mask

    # Undistort images
    proj_density = [
                    cv2.undistort(x, cc['Intrinsics'], cc['Distortion'])
                    for (x, cc) in zip(proj_density, camCal)
                    ]

    # Swap axes to match ASTRA Toolbox's preferred order
    proj_density = np.swapaxes(proj_density, 0, 1)

    return proj_density


def showSprayPB(state, pb, Re, units, cmap):
    # Unpack inputs
    Re = Re[state]
    pb = pb[state]

    # Setup figure
    fig = plt.figure(figsize=(10,6))
    fig.subplots_adjust(top=0.99, bottom=0.4)
    fig.suptitle(f'Projection Images [{units}] - $Re$ = {Re}')

    ax = ImageGrid(
                   fig,
                   111,
                   nrows_ncols=(1,3),
                   axes_pad=0.5,
                   cbar_location='right',
                   cbar_mode='each',
                   cbar_size=0.15,
                   cbar_pad=0.1,
                   cbar_set_cax=True
                   )

    # Plot spray images
    im = [
          axi.imshow(x, cmap=cmap)
          for axi, x in zip(ax, pb.swapaxes(0, 1))
          ]

    # Show colorbars
    [ax.cbar_axes[i].colorbar(im[i]) for i in range(3)]

    # Set subplot title
    [axi.set_title(f'Camera {i+1}') for i, axi in enumerate(ax)]

    # Turn off axes
    [axi.set_axis_off() for axi in ax]

    return
