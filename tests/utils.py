import numpy as np


def quantitative_match(python_logs, emergent_logs, limit=-1, check=True, rtol=1e-05, atol=0):
    """Check that python and emergent data match quantitatively

    Parameters:
        limit    consider only a prefix of `limit` length of the data (default -1, for full data).
        check    the function will return the boolean "result of the match and `check`" (default True).
                 Used to carry over results from previous checks.
        rtol     relative tolerance. See `numpy.allclose` function.
        atol     absolute tolerance. See `numpy.allclose` function. Be
    """

    check = check
    for name in python_logs.keys():
        for t, (py, em) in enumerate(list(zip(python_logs[name], emergent_logs[name]))[:limit]):
            if not np.allclose(py, em, rtol=rtol, atol=atol):
                print('{}:{:2d} [py] {} != {} [em] diff={}'.format(
                       name, t,      py,  em,        py-em))
                check = False
    return check
