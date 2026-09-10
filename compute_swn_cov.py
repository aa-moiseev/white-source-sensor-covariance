"""
**Compute sensor covariance matrix corresponding to white uncorrelated noise at
the brain sources level**
"""
import numpy as np
from nearest_pos_def import nearestPD, isPD

def compute_swn_cov(fwd, data_cov, *, tol = 1e-2, rcond = 1e-10, max_rank = None):
    """Based on the forward solutions, construct sensor-level noise covariance
    matrix assuming white, randomly oriented uncorrelated brain sources. Also
    calculate (pseudo-)inverse of the data covariance and sensor level SNR.

    The basic expression for noise covariance is:

    `cov0 = const * SUM(i=1 to Nsrc){Hx Hx' + Hy Hy' + Hz Hz'}`

    where `Hx,y,z(i)` are forward solutions for i-th source with corresponding
    orientations, and `const` is defined so as data_cov - noise_cov is non-negative.

    For degenerate data_cov the noise_cov should also be degenerate with a 
    range subspace coinciding with the range of the data_cov. In this case the
    above expression should be replaced with

    `cov = P * cov0 * P`

    where `P = data_cov * pinv(data_cov)` is a projector on the `range(data_cov)`.

    The trace of the noise_cov is maximized while keeping the difference
    `data_cov - noise_cov` non-negative. The tol parameter defines how close to
    the upper boundary one should get.

    This function also upgrades both calculated inverse of data covariance, and
    the noise covariance to the nearest positive definite matrices. This is done
    to avoid potential problems with getting non-positive matrices and complex
    eigenvalues due to rounding errors, when truly degenerate matrices are involved.

    Args:
        fwd (Forward): mne Python forward solutions class
        data_cov (ndarray): nchan x nchan data covariance matrix, possibly with environmental
            or empty room covariance subtracted. MUST BE POSITIVE-DEFINITE
        tol (float > 0): tolerance in finding the noise_cov trace.
        rcond (float > 0): singular values less or equal to max(sing val) * rcond will
            be dropped
        max_rank(int>0 | None): if specified, sets the upper bound for rank of
            the covariance. The upper bound may be known if data is max-filtered,
            or for EEG data depending on the reference settings and projectors used.
            It is assumed that any computed rank larger than the max_rank will be
            incorrect.

    Returns:
        cov (ndarray): (nchan x nchan) noise cov matrix, such that the difference
            data_cov - noise_cov is non-negatively defined
        inv_cov (ndarray): (nchan x nchan) (pseudo-) inverse of the data cov
        rank (int): rank of the data covariance
        pz (float): psedo-Z = SNR + 1 of the data; `pz=tr(data_cov)/tr(noise_cov)`

    """
    # Reduce all calcs to a full-rank subspace of the data_cov
    # U, Vh = nchan x nchan, S = (nchan,). In fact in this case Vh' = U
    U, S, Vh = np.linalg.svd(data_cov, full_matrices=False, hermitian=True)

    # Drop close to 0 singular values and corresponding columns of U. Note that all S[i]
    # are positive and sorted in decreasing order
    t = rcond * S[0]
    U = U[:, S > t]
    rank = U.shape[1]	# The actual rank of the data covariance

    if (max_rank is not None):
        if rank > max_rank:
            rank = max_rank
            U = U[:, :rank]

    H = fwd['sol']['data']	# Should be nchan x (3*nsrc) matrix

    # Reduced unnormalized noise covariance
    uH = U.T @ H
    unoise_cov = uH @ uH.T	# unoise_cov = U' H H' U

    # Reduced data covariance
    udata_cov = np.diag(S[:rank])    # udata_cov = U' U S U' U = S

    # (Pseudo-) inverse of the covariance
    # Make it pos def instead fully degenerate
    inv_cov = nearestPD(U @ np.diag(1./S[:rank]) @ U.T)

    # Initially, normalize it with the trace of data_cov
    pwr = np.trace(udata_cov)
    unoise_cov = (pwr / np.trace(unoise_cov)) * unoise_cov

    upper = pwr    # Current upper value of the trace (not pos def)
    lower = 0.     # Current lower value of the trace (already pos def)
    tr = pwr       # New value of the trace
    tr0 = tr       # Old value of the trace
 
    while True:
        if isPD(udata_cov - unoise_cov):
            # Set current value of the trace as lower bound; move the next
            # value to try up by half distance to upper bound
            lower = tr
            tr = lower + (upper - lower)/2
            is_pd = True        # Flag current difference (data - noise) as pos def
        else:
            # Overshoot happened. Set current value of the trace as new upper bound;
            # move next value to try half distance down
            upper = tr
            tr = upper - (upper - lower)/2
            is_pd = False       # Flag current difference (data - noise) as pos def

        assert upper > lower	# Just a sanity check; should always be true

        # Scale the noise cov to the new setting to try
        ratio = tr/tr0
        unoise_cov = ratio * unoise_cov
        
        # Save the new value of the trace of the noise cov
        tr0 = tr

        if is_pd and (np.abs(ratio - 1) < tol):
            break   # We exit after moving a bit up from the verified PD state. Even if
                    # we overshoot - this is within tolerance, and will be taken
                    # care of by applying nearestPD to the result

    # Project results back to the original sensor space
    noise_cov = nearestPD(U @ unoise_cov @ U.T)
    pz = pwr / tr
        
    return noise_cov, inv_cov, rank, pz

