from .algebra import Variable
from fractions import Fraction
from io import StringIO
from typing import Any, Sequence
from numbers import Real
import math, numpy as np

__all__ = ['format_table', 'Tableau']

def format_table(
        data: Sequence[Sequence[str]],
        shape: tuple[int, int],
        sep: str) -> str:
    '''
    Formats a tabular data into tab-separated string.

    Parameters
    ----------
    data:
        A 2-D sequence of strings.
    shape:
        Shape of `data` (H, W).
    sep:
        Separator character inserted between lines.
    
    Returns
    -------
    string:
        Formatted string.
    '''
    result = StringIO()
    column_widths = []
    for col in range(shape[1]):
        width = 0
        for row in range(shape[0]):
            if width < len(data[row][col]):
                width = len(data[row][col])
        column_widths.append(width // 8 + 1)
    for row in data:
        for value, width in zip(row, column_widths):
            tabs_needed = math.ceil(width - len(value) / 8)
            result.write(value)
            result.write('\t' * tabs_needed)
        result.write('\n')
        result.write(8 * sum(column_widths) * sep)
        result.write('\n')
    return result.getvalue()

class Tableau:
    '''
    A tableau in a Simplex Algorithm step.

    Example
    -------
    >>> import numpy as np
    >>> # fake data, not a valid problem
    >>> data = np.eye(2) / 5
    >>> sigma = rhs = np.ones(2) / 3
    >>> z = 0.5
    >>> bvs = vs = ['x1', 'x2']
    >>> t = Tableau(data, sigma, rhs, z, vs, bvs)
    >>> print(t)
    BV      x1      x2      RHS     
    x1      1/5     0       1/3
    x2      0       1/5     1/3
            1/3     1/3     z-1/2
    '''

    data: np.ndarray
    'Data matrix.'
    sigma: np.ndarray
    'Sigma matrix.'
    rhs: np.ndarray
    'Right-hand side matrix.'
    z: Real
    'Negated z value.'
    variables: list[Variable]
    'List of variables.'
    bases: list[Variable]
    'List of base variables.'
    precision: int | None
    '''
    The precision of the result when displayed.
    If `None`, display using closest fraction.
    '''

    def __init__(self, 
                 data: np.ndarray, 
                 sigma: np.ndarray, 
                 rhs: np.ndarray,
                 z: Real, 
                 vs: list[Variable], 
                 bvs: list[Variable], 
                 precision: int | None = None):
        '''
        Initializes a tableau.

        Parameters
        ----------
        data, sigma, rhs, z:
            Matrices in the simplex step.
        vs:
            List of variables.
        bvs:
            List of base variables.
        precision:
            The precision of the result when displayed.
            If `None`, display using closest fraction.
        '''
        self.data = data
        self.sigma = sigma
        self.rhs = rhs
        self.z = z
        self.variables = vs
        self.bases = bvs
        self.precision = precision
    def _to_string(self, n: Any, plus_sign: bool = False) -> str:
        '''
        Converts an object to string using `self.precision`.

        Parameters
        ----------
        n:
            The object to convert.
            If numeric, use `self.precision`.
        plus_sign:
            If `True`, adds a `+` sign if greater than 0.
        
        Returns
        -------
        string:
            A string representation of `n`.
        '''
        if isinstance(n, (int, float)):
            if self.precision is not None and self.precision >= 0:
                return format(n, (plus_sign * '+') + f'.{self.precision}g')
            else:
                n = Fraction(n).limit_denominator()
                if n >= 0 and plus_sign:
                    return '+' + str(n)
                else:
                    return str(n)
        return str(n)
    def to_frame(self) -> list[list[str]]:
        '''
        Converts this object to a 2-D array of formatted strings.

        Returns
        -------
        table:
            A table of strings, useful in `format_table`.
        '''
        data = [
            ['BV'] + 
            list(map(self._to_string, self.variables)) + 
            ['RHS'],
        ]
        for i, row in enumerate(self.data):
            data.append(
                [self._to_string(self.bases[i])] + 
                list(map(self._to_string, row)) + 
                [self._to_string(self.rhs[i])]
            )
        data.append(
            [''] + 
            list(map(self._to_string, self.sigma)) + 
            [f'z{self._to_string(self.z, True)}']
        )
        return data
    def display(self, sep: str = '-') -> str:
        '''
        Converts this object to table format.

        Parameters
        ----------
        sep:
            Separator inserted between lines.

        Returns
        -------
        string:
            Formatted table.
        '''
        data = self.to_frame()
        return format_table(data, (self.data.shape[0]+2, self.data.shape[1]+2), sep)
    def __str__(self):
        return self.display()
    def __repr__(self):
        return 'Tableau({}, {})'.format(*self.data.shape)