"""This module devises metrics to compare estimates from analysis models."""

import numpy as np
import pandas as pd

def raw_bias(Q_bar, Q):
    """Calculate raw bias between coefficients Q and actual Q.

    Q_bar can be one estimate (scalar) or a vector of estimates. This equation
    subtracts the expected Q_bar from Q, element-wise. The result is the bias
    of each coefficient from its true value.

    Args:
        Q_bar (number, array): single estimate or array of estimates.
        Q (number, array): single truth or array of truths.

    Returns:
        scalar, array: element-wise difference between estimates and truths.

    Raises:
        ValueError: Shape mismatch
        ValueError: Q_bar and Q not the same length
    """

    # handle errors first
    shape_err = "Q_bar & Q must be scalars or vectors of same length."
    if isinstance(Q_bar, pd.DataFrame):
        s = len(Q_bar.shape)
        if s != 1:
            raise ValueError(shape_err)

    if isinstance(Q, pd.DataFrame):
        s = len(Q.shape)
        if s != 1:
            raise ValueError(shape_err)

    if len(Q_bar) != len(Q):
        raise ValueError(shape_err)

    # convert any lists to ensure element-wise performed
    if isinstance(Q_bar, (tuple, list)):
        Q_bar = np.array(Q_bar)

    if isinstance(Q, (tuple, list)):
        Q = np.array(Q)

    # perform element-wise subtraction
    rb = Q_bar - Q
    return rb

def percent_bias(Q_bar, Q):
    """Calculate precent bias between coefficients Q and actual Q.

    Q_bar can be one estimate (scalar) or a vector of estimates. This equation
    subtracts the expected Q_bar from Q, element-wise. The result is the bias
    of each coefficient from its true value. We then divide this number by
    Q itself, again in element-wise fashion, to produce % bias.

    Args:
        Q_bar (number, array): single estimate or array of estimates.
        Q (number, array): single truth or array of truths.

    Returns:
        scalar, array: element-wise difference between estimates and truths.

    Raises:
        ValueError: Shape mismatch
        ValueError: Q_bar and Q not the same length
    """
    # calling this method will validate Q_bar and Q
    rb = raw_bias(Q_bar, Q)

    # convert Q if necessary. must re-perform operation
    if isinstance(Q, (tuple, list)):
        Q = np.array(Q)

    pct_bias = 100 * (abs(rb)/Q)
    return pct_bias
