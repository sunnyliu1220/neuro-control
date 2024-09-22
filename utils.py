import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def get_ranked_indices(r,thr = 0.1):
    '''
    Returns indices sorted based on the maximum firing rate of each neuron.
    Inputs:
    - r: firing rates (T, n_rec)
    '''
    n_cell = r.shape[1]
    max_active = np.zeros(n_cell)
    for i in range(n_cell):
        temp = r[:,i]
        max_ind = np.argmax(temp)
        if max(temp) < thr:
            max_active[i] = 0
        else:
            max_active[i] = max_ind
        
    inds_all = np.argsort(max_active)
    return inds_all

def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    Source: https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)

def sample_slds(slds, y_init, T_total):
    '''
    Sample from an SLDS model for a total of T_total time steps.
    SLDS (of which LDS is a special case) takes z, x, y as the input state
    z is the discrete state, x is the latent continuous state, and y is the observation
    We set z to 0, x to the best estimate of the latent state, and y to anything

    First, we compute the approximate posterior over continuous
    states for the new data under the current model parameters.
    We do this because we have a probability distribution over x, not a deterministic one.
    Inputs:
    - slds: SLDS model
    - y_init: initial state (T_init, n_obs)
    - T_total: total number of time steps
    '''
    T_init = y_init.shape[0]
    elbos_pred, posterior = slds.approximate_posterior(y_init,
                                                method="laplace_em",
                                                variational_posterior="structured_meanfield",
                                                num_iters=50)
    # Get the posterior mean of the continuous states, x
    x_init = posterior.mean_continuous_states[0]
    print (f'x_init_test has shape: {x_init.shape}')
    z_init= np.zeros(T_init, dtype=int)
    states, emissions = slds.sample(T_total-T_init, prefix=(z_init, x_init, y_init))
    states = np.concatenate((x_init, states), axis=0)
    emissions = np.concatenate((y_init, emissions), axis=0)
    emissions_smoothed = slds.smooth(states, emissions)

    return elbos_pred, emissions_smoothed

def finite_MAP(alpha, T, observations, phi, rho):
    '''
    Finite time MAP estimate of the dynamics matrix according to the paper.
    '''
    D_s = observations.shape[1] # Student number of neurons
    first_sum = np.zeros((D_s, D_s))
    for t in np.arange(0, T):
        first_sum += np.outer(observations[t]-(1-alpha)*observations[t-1], phi(observations[t-1]))
    second_sum = np.zeros((D_s, D_s))
    for t in np.arange(0, T):
        second_sum += np.outer(phi(observations[t-1]), phi(observations[t-1]))
    A_T = alpha / T * first_sum @ np.linalg.inv(second_sum * alpha**2 / T + rho * np.eye(D_s))
    return A_T