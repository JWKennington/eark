"""Testing utilities
"""
import numpy as np

DEFAULT_PRECISION = 3
DEFAULT_MAX_LINE_WIDTH = 500


def assert_array_approx_equal(res: np.ndarray, desired: np.ndarray, significant: int = DEFAULT_PRECISION):
    try:
        for r, d in zip(res.ravel(), desired.ravel()):
            np.testing.assert_approx_equal(actual=r, desired=d, significant=significant)
    except AssertionError as e:
        res_str = np.array2string(a=res, max_line_width=DEFAULT_MAX_LINE_WIDTH, precision=significant, separator=',',
                                  formatter={'float': lambda x: np.format_float_scientific(x, precision=significant)})
        des_str = np.array2string(a=desired, max_line_width=DEFAULT_MAX_LINE_WIDTH, precision=significant, separator=',',
                                  formatter={'float': lambda x: np.format_float_scientific(x, precision=significant)})
        msg = '\nArrays not approximately equal.\nExpected:\n{}\nGot:\n{}'.format(des_str, res_str)
        raise AssertionError(msg) from e
