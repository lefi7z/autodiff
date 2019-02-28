"""Toolbox from autodiff.

"""
from .core import d

__all__ = ['newton']


class Result(dict):

    def __repr__(self):
        return '\n'.join(sorted("%s: %s" % (str(k), str(v)) 
                              for k, v in self.items()))

    def __getattr__(self, key):
        return self[key]

    def __bool__(self):
        if 'success' in self:
            return self['success']
        else:
            return False


def prepare(fun):
    nfev = 0
    def wrapped(x):
        nonlocal nfev
        nfev += 1
        return nfev, fun(x)
    return wrapped


def newton(fun, x0, atol=1e-8, maxiter=200):
    """Super-simple implementation of Newton's secant method.

    Examples:
    >>> f = lambda x: 3 + x - x**3
    >>> x0 = 3
    >>> res = newton(f, x0, atol=1e-1)
    >>> res.x
    1.6807929652716966
    >>> res.nit
    3

    get a higher precision in linear time:
    >>> res = newton(f, x0, atol=1e-10)
    >>> res.nit
    6

    the start value matters:
    >>> x0 = -2
    >>> res = newton(f, x0, atol=1e-2)
    >>> res.nit
    20

    limit the iterations and test for success:
    >>> res = newton(f, x0, atol=1e-2, maxiter=15)
    >>> res
    message: max iterations reached.
    nfev: 30
    nit: 15
    success: False
    x: 4.603428431640553
    >>> if not res: print('fail..')
    fail..
    
    """
    count = 0
    success = False
    _fprime = d(fun)
    _fun = prepare(fun)
    while count < maxiter:
        nfev, f_x0 = _fun(x0)
        if abs(f_x0) < atol:
            msg = "converged."
            success = True
            break
        count += 1
        x0 += -f_x0 / _fprime(x0)
    else:
        msg = "max iterations reached."

    return Result(message=msg, success=success, 
                  nit=count, nfev=2*nfev, x=x0)

