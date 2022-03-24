import copy
import cv2
import numpy as np
import pyswarms as ps
import time

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from numpy.random import uniform
from pyswarms.backend.operators import (
                                        compute_pbest,
                                        compute_objective_function as comp_obj
                                        )

from recon_utilities import (
                             getSystemMatrix,
                             MLOS
                             )


def geomVec(camCal, det_size, view):
    """
    Extracting all parameters from the camera calibration routine that will be
    used for geometry refinement as a vector.

    :param list(dict) camCal: Camera calibration parameters.
    :param list(float) det_size: Detector size for each camera [mm/px].
    :param int view: Which calibration plate view to use to define the world.
    :returns: Vectorized geometry parameters to be used in PSO routine.
    :rtype: numpy.array(numpy.float32)
    """
    # Get all rotation vectors
    Rv = np.array([x['Rotation'][view] for x in camCal]).flatten()

    # Convert to degrees
    Rv = np.rad2deg(Rv)

    # Get all translation vectors in mm
    Tv = np.array([x['Translation'][view] for x in camCal]).flatten()

    # Get all focal lengths in px
    fl = np.array([
                   x['Intrinsics'][[0,1], [0,1]].mean()
                   for x in camCal
                   ]).flatten()

    # Get all principal points in px
    pp = np.array([x['Intrinsics'][:2, -1] for x in camCal]).flatten()

    # Collate all lists
    gVec = np.concatenate((Rv, Tv, fl, pp, det_size))

    return gVec


def geomDict(vec):
    """
    Convert the vectorized geometry into a more easily read dict. The dict
    format is also later used in the system matrix setup.

    :param numpy.array(numpy.float32) vec: Geometry vector.
    :returns: Geometry dict for each line of sight.
    :rtype: list(dict)
    """
    # Rotations
    Rv = np.deg2rad(vec[:9]).reshape((3, 3))

    # Translations
    Tv = vec[9:18].reshape((3, 3))

    # Focal lengths
    fl = vec[18:21]

    # Principal points
    pp = vec[21:27].reshape((3, 2))

    # Detector size
    det_size = vec[27:]

    # Initialize list of dicts
    geom = [None] * 3

    # Run through each of the 3 lines of sight
    for i in range(3):
        R_i = cv2.Rodrigues(Rv[i])[0]
        T_i = Tv[i]

        fl_i = fl[i]
        pp_i = pp[i]

        det_size_i = det_size[i]

        # Get source position in world coordinates
        # This is just the inverse of the extrinsics!
        source_world = (-R_i.T @ T_i)

        # Get detector principal point position in source coordinates
        # This is just a focal length away from the source
        pp_source = np.array([0, 0, fl_i])*det_size_i

        # Now find the center of the detector in source coordinates
        detCenter_source = pp_source + np.append(256-pp_i, 0)*det_size_i

        # And finally the center of the detector in world coordinates
        # Same transformation process as above for the source
        detCenter_world = R_i.T @ (detCenter_source - T_i)

        # Get detector orientation in **detector coordinates**
        # U: X direction (columns of array!)
        # V: Y direction (rows of array!)

        # Detector origin: (U, V) = (0, 0)
        detOrigin = np.array([0, 0])

        # Get the coordinates defining the +U and +V unit directions
        uPos = np.array([1, 0])
        vPos = np.array([0, 1])

        # Convert to source coordinates ("intrinsics")
        detOrigin_source = pp_source + np.append(detOrigin-pp_i, 0)*det_size_i
        uPos_source = pp_source + np.append(uPos-pp_i, 0)*det_size_i
        vPos_source = pp_source + np.append(vPos-pp_i, 0)*det_size_i

        # Convert to world coordinates (extrinsics)
        detOrigin_world = R_i.T @ (detOrigin_source - T_i)
        uPos_world = R_i.T @ (uPos_source - T_i)
        vPos_world = R_i.T @ (vPos_source - T_i)

        # And finally the unit vectors defining the U and V axes
        uVec_world = uPos_world - detOrigin_world
        vVec_world = vPos_world - detOrigin_world

        # Collect all positions/directions for this line of sight into a dict
        geom[i] = {
                   'Anode': source_world.tolist(),
                   'Detector': detCenter_world.tolist(),
                   'U Axis': uVec_world.tolist(),
                   'V Axis': vVec_world.tolist(),
                   'Detector Size': det_size_i
                   }

    return geom


def showGeom(geom):
    """
    Top-down visualization of the line of sight geometries.

    :param list(dict) geom: Geometry dict for each line of sight.
    """

    # Setup figure
    fig = plt.figure()

    # Color of each line of sight
    colors = ['r', 'g', 'b']

    for i in range(3):
        # Plot anode location
        plt.scatter(
                    *geom[i]['Anode'][0:3:2],
                    color=colors[i],
                    marker='s'
                    )

        # Plot detector location
        plt.scatter(
                    *geom[i]['Detector'][0:3:2],
                    color=colors[i],
                    marker='d'
                    )

        # Connect the anode/detector pair
        plt.plot(
                [geom[i]['Anode'][0], geom[i]['Detector'][0]],
                [geom[i]['Anode'][-1], geom[i]['Detector'][-1]],
                color=colors[i],
                linestyle='--')

    # Create custom legend
    leg_el = [
              Line2D([0], [0], 0, '', 'k', 's', mfc='w', label='Anode'),
              Line2D([0], [0], 0, '', 'k', 'd', mfc='w', label='Detector')
              ]
    plt.legend(handles=leg_el)

    # Labels
    plt.xlabel('X [mm]')
    plt.ylabel('Z [mm]')
    plt.title('Initial Geometry Setup')

    return


def compareProj(init, reproj, title):
    """
    Visualization showing the differences map between the binarized projection
    and reprojection dataset. Yellow = extra pixels (structural information
    not present in the original projections), red = missing pixels (structural
    information not present in the reprojections).

    :param numpy.ndarray(numpy.float32) init: Initial projection data.
    :param numpy.ndarray(numpy.float32) reproj: Reprojection data.
    :param str title: String designating before/after PSO refinement and cost. 
    """
    proj_diff = (reproj > 0).astype(float) - (init > 0).astype(float)
    proj_diff[proj_diff == -1] = np.nan
    cmap = copy.deepcopy(plt.cm.viridis)
    cmap.set_bad(color='red')

    # Setup figure
    fig, ax = plt.subplots(1, 3, figsize=(10,6))
    fig.subplots_adjust(top=0.99, bottom=0.4)
    fig.suptitle(f'Projection Difference - {title}')

    ax[0].imshow(proj_diff[:,0], cmap=cmap)
    ax[0].axhline(y=400, color='w', linestyle='--')
    ax[1].imshow(proj_diff[:,1], cmap=cmap)
    ax[1].axhline(y=400, color='w', linestyle='--')
    ax[2].imshow(proj_diff[:,2], cmap=cmap)
    ax[2].axhline(y=400, color='w', linestyle='--')

    # Set subplot title
    [axi.set_title(f'Camera {i+1}') for axi, i in zip(ax, range(3))]

    # Turn off axes
    [axi.set_axis_off() for axi in ax]

    # Create custom legend
    leg_el = [
              Patch(fc=cmap(256), ec=cmap(256), label='Extra'),
              Patch(fc='red', ec='red', label='Missing'),
              Line2D([0], [0], color='white', linestyle='--', label='Cut-off')
              ]
    ax[0].legend(handles=leg_el, loc='upper left')

    return


def setupConstraints(vec, errCal, view):
    """
    Calculation of bounds for a constrained objective function for better
    convergence in the PSO routine. Based on the errCal variable which defines
    the calibration parameter uncertainties.

    :param numpy.array(numpy.float32) vec: Geometry vector.
    :param list(dict) errCal: Standard deviation values from the calibration.
    :param int view: Which calibration plate view to use to define the world.
    :returns: Tuple of lower and upper bounds for constrained PSO.
    :rtype: (numpy.array(numpy.float32), numpy.array(numpy.float32))
    """
    # Unpack standard deviations from calibration process
    intErr = [x['Intrinsics StDev'][:4] for x in errCal]
    extErr = [x['Extrinsics StDev'].reshape((3, -1))[view] for x in errCal]

    # Rotations
    Rv = vec[:9]
    Rstd = np.array([np.rad2deg(x[:3]) for x in extErr]).ravel()
    lowerR = Rv - 3*Rstd
    upperR = Rv + 3*Rstd

    # Translations
    Tv = vec[9:18]
    Tstd = np.array([x[3:] for x in extErr]).ravel()
    lowerT = Tv - 3*Tstd
    upperT = Tv + 3*Tstd

    # Focal lengths
    fl = vec[18:21]
    flstd = np.array([np.mean(x[:2]) for x in intErr])
    lowerfl = fl - 3*flstd
    upperfl = fl + 3*flstd

    # Principal points
    pp = vec[21:27]
    ppstd = np.array([x[2:] for x in intErr]).ravel()
    lowerpp = pp - 3*ppstd
    upperpp = pp + 3*ppstd

    # Detector size
    # Standard deviations estimated from formula d = w/i
    # std(d) = d * sqrt((std_w)/w)**2 + (std_i)/i)**2)
    # d: detector size, w: detector length (mm), i: detector length (px)
    # std_w: std of w (estimated), std_i: std of i (estimated)
    # https://en.wikipedia.org/wiki/Propagation_of_uncertainty
    # DOI: 10.4135/9781412983341.n4
    d = vec[27:]
    w = np.array([50.8, 50.8, 50.8])
    std_w = np.array([2, 2, 2])
    i = np.array([459, 460, 471])
    std_i = np.array([5, 5, 5])
    dstd = d * np.sqrt((std_w/w)**2 + (std_i/i)**2)
    lowerd = d - 3*dstd
    upperd = d + 3*dstd

    # Collate all lists
    lowercon = np.concatenate((lowerR, lowerT, lowerfl, lowerpp, lowerd))
    uppercon = np.concatenate((upperR, upperT, upperfl, upperpp, upperd))

    return (lowercon, uppercon)


def getError(W, proj):
    """
    Objective function for single particle. Calculates the number of "missing
    pixels" in the reprojections--minimization of this leads to a better set
    of geometry parameters for an improved reconstruction.

    :param OpTomo W: System matrix from ASTRA.
    :param numpy.ndarray(numpy.float32) proj: Initial projection data.
    :returns: Error value defining the particle's solution distance.
    :rtype: float
    """
    # Calculate reprojection using current particle's system matrix W
    _, reproj = MLOS(W, proj, 'proj', 3)

    # Binarize the projections (only want the structural differences)
    # Taking care of invalid values (NaN) through out/where
    binReproj = np.greater(
                           reproj,
                           0,
                           out=np.zeros(reproj.shape, dtype=np.float32),
                           where=np.isfinite(reproj)
                           )
    binProj = np.greater(
                         proj,
                         0,
                         out=np.zeros(proj.shape, dtype=np.float32),
                         where=np.isfinite(reproj)
                         )

    # Count total number of pixels "missing" in the reprojection
    # Camera 3 sees more of the spray, so cropping to slightly above
    error = np.count_nonzero((binReproj-binProj)[:400,:,:] < 0)

    return error


def objective(x, proj):
    """
    Handle to the getError() objective function for all of the PSO particles.

    :param ss x: Vector of PSO particles.
    :param numpy.ndarray(numpy.float32) proj: Initial projection data.
    :returns: Vector of objective function evaluations for all particles.
    :rtype: numpy.array(numpy.float32)
    """
    proj = copy.deepcopy(proj)
    errors = [getError(getSystemMatrix(geomDict(y)), proj) for y in x]

    return np.array(errors)


def optimize(
             projData,
             maxiters,
             swarm_size,
             dim,
             constraints,
             init_pos,
             oh_strategy,
             start_opts,
             end_opts
             ):
    """
    Main function that handles the PSO routine. Bits and pieces taken from the
    official PySwarms documentation (https://pyswarms.readthedocs.io/).

    :param numpy.ndarray(numpy.float32) projData: Initial projection data.
    :param int maxiters: Maximum number of iterations to run.
    :param int swarm_size: Number of particles to use.
    :param int dim: Number of parameters to refine.
    :param constraints: Lower and upper bounds for the refinement parameters.
    :ptype constraints: (numpy.array(float), numpy.array(float))
    :param numpy.array(float) init_pos: Initial particle positions.
    :param dict oh_strategy: Update strategies for the hyper-parameters.
    :param dict start_opts: Initial hyper-parameter values.
    :param dict end_opts: Ending hyper-parameter values.
    :returns: Best cost, corresponding positions, and PSO object.
    :rtype: (float, float, ps.single.LocalBestPSO)
    """
    init_swarm = np.vstack((
                            init_pos,
                            np.array([
                                      uniform(constraints[0], constraints[1])
                                      for i in range(swarm_size-1)
                                      ])
                            ))

    opt = ps.single.LocalBestPSO(
                                 n_particles=swarm_size,
                                 dimensions=dim,
                                 bounds=constraints,
                                 init_pos=init_swarm,
                                 options=start_opts,
                                 oh_strategy=oh_strategy
                                 )

    swarm = opt.swarm
    opt.bh.memory = swarm.position
    opt.vh.memory = swarm.position
    swarm.pbest_cost = np.full(opt.swarm_size[0], np.inf)

    # Number of neighbors to compare against for local best PSO
    k = start_opts['k']

    # Minkowski p-norm (1 = L1 norm, 2 = L2 norm)
    p = start_opts['p']

    print(time.ctime())
    start_time = time.time()
    counter = 0
    for i in range(maxiters):
        # Compute cost for current position
        swarm.current_cost =  comp_obj(
                                       swarm=swarm,
                                       objective_func=objective,
                                       pool=None,
                                       **{'proj': projData}
                                       )

        # Compute personal best
        swarm.pbest_pos, swarm.pbest_cost = compute_pbest(swarm)

        # Perform options update
        swarm.options = opt.oh(
                               opt.options,
                               iternow=i,
                               itermax=maxiters,
                               end_opts=end_opts
                               )

        # Perform velocity and position updates
        swarm.velocity = opt.top.compute_velocity(
                                                  swarm,
                                                  opt.velocity_clamp,
                                                  opt.vh,
                                                  opt.bounds
                                                  )
        swarm.position = opt.top.compute_position(
                                                  swarm,
                                                  opt.bounds,
                                                  opt.bh
                                                  )

        # Display status
        elapsed_time = (time.time() - start_time)/60
        time_left = maxiters*(elapsed_time/(i+1)) - elapsed_time
        eta = time.time() + (time_left*60)
        swopt = swarm.options.copy()
        print(f'Iteration: {i} | Options: ' +
              'c1={c1:0.3f}, c2={c2:0.3f}, w={w:0.3f}'.format(**swopt) +
              f' | Cost: {swarm.best_cost:0.1f}')
        print(f'Elapsed time: {elapsed_time:0.2f} min | '
              f'Time left: {time_left:0.0f} min | ETA: {time.ctime(eta)}')

    # Obtain the final best_cost and the final best_position
    final_best_cost = swarm.best_cost.copy()
    final_best_pos = swarm.pbest_pos[swarm.pbest_cost.argmin()].copy()

    return final_best_cost, final_best_pos, opt
