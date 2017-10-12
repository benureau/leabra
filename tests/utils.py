import numpy as np
import color


def quantitative_match(python_logs, emergent_logs, limit=-1, check=True, rtol=1e-05, atol=0,
                       verbose=0, postprocessing=None, prefix='', suffix=''):
    """Check that python and emergent data match quantitatively

    Parameters:
        limit    consider only a prefix of `limit` length of the data (default -1, for full data).
        check    the function will return the boolean "result of the match and `check`" (default True).
                 Used to carry over results from previous checks.
        rtol     relative tolerance. See `numpy.allclose` function.
        atol     absolute tolerance. See `numpy.allclose` function.
        verbose  0: no output. 1: output if error. 2: output regardless.
    """
    check = check
    for name in python_logs.keys():
        for t, (py, em) in enumerate(list(zip(python_logs[name], emergent_logs[name]))[:limit]):
            if postprocessing is not None:
                py, em = postprocessing(py, em)
            local_check = np.allclose(py, em, rtol=rtol, atol=atol)
            if verbose >= 2 or (verbose >= 1 and not local_check):
                same = np.allclose(py, em, rtol=0, atol=0)
                if same:
                    sign_text = '==', color.dye_out('same', 'green')
                elif local_check:
                    sign_text = '~=', color.dye_out(' ok ', 'green')
                else:
                    sign_text = '!=', color.dye_out('fail', 'bred')

                try:
                    diff_text = '' if same else 'diff={:+.2e}'.format(py - em)
                    print('{}{}:{:2d} [{}] [py] {:12.10f} {} {:12.10f} [em] {} {}'.format(prefix,
                           name, t, sign_text[1], py, sign_text[0], em, diff_text, suffix))
                except TypeError:
                    diff_text = '' if same else 'diff={}'.format(py - em)
                    print('{}{}:{:2d} [{}] [py] {} {} {} [em] {} {}'.format(prefix,
                           name, t, sign_text[1], py, sign_text[0], em, diff_text, suffix))
            check = check and local_check
    return check
