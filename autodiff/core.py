"""Automatic differentiation module.

"""
import typing

import numpy as np


__all__ = ['d', 'val', 'Num', 'Var']


def val(var):
    """Return the value part of `var`."""
    if isinstance(var, Num):
        return var.x

    elif isinstance(var, (float, int, np.ndarray)):
        return var

    else:
        raise ValueError("Cannot convert %s to value!" % str(var))


def d(var, n=1):
    """Return the `n`th derivative of `var`.
    
    `var` may be a function or a number.
    """
    if n > 1:
        return d(var, n-1)

    if isinstance(var, Num):
        return var.dx

    if isinstance(var, typing.Callable):
        def fprime(x):
            return var(Num(x, 1.0)).dx
        return fprime

    else:
        return 0.


class Num:
    """Dual number class that saves value and derivative.
    
    >>> Var(4) + 3
    7
    >>> d(Var(4))
    1.0
    >>> v = Var(7)
    >>> 3*v + v**2
    70
    >>> d(3*v + v**2)
    17.0
    >>> u = 3*v + v**2
    >>> u.x, u.dx
    (70, 17.0)
    >>> f = lambda x: 3 + x - x**3
    >>> fprime = d(f)
    >>> fprime(7)
    -146.0

    """
    def __init__(self, x, dx):
        self.x = x
        self.dx = dx

    def __float__(self):
        return self.x

    def __add__(self, other):
        x, y, dx, dy = val(self), val(other), d(self), d(other)
        x, dx = x + y, dx + dy

        return Num(x, dx)

    __radd__ = __add__

    def __sub__(self, other):
        x, y, dx, dy = val(self), val(other), d(self), d(other)
        x, dx = x - y, dx - dy

        return Num(x, dx)

    def __rsub__(self, other):
        return -self + other

    def __neg__(self):
        return -1*self

    def __mul__(self, other):
        x, y, dx, dy = val(self), val(other), d(self), d(other)
        x, dx = x * y, dx * y + x * dy

        return Num(x, dx)

    __rmul__ = __mul__

    def __truediv__(self, other):
        x, y, dx, dy = val(self), val(other), d(self), d(other)
        if y == 0.:
            return Num(np.inf, np.inf)

        x, dx = x / y, (dx*y - x*dy) / y**2

        return Num(x, dx)

    def __rtruediv__(self, other):
        return Num.__truediv__(other, self)

    def __pow__(self, other):
        x, y, dx, dy = val(self), val(other), d(self), d(other)
        x, dx = x**y, y * x**(y-1)

        return Num(x, dx)

    def __repr__(self):
        return str(val(self))


class Var(Num):
    """Shorthand for Num(x, 1.0)"""

    def __init__(self, x):
        super().__init__(x, 1.0)

