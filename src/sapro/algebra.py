from .error import InvalidSlack
from numbers import Real
from typing import Any, Generator, TypeVar, Literal, Mapping

__all__ = ['Variable', 'VariablePool', 'Expression', 'Constraint', 'OperatorType']

_T = TypeVar('_T')

class Variable:
    '''
    A variable used in target / constraint definition.

    Example
    -------
    >>> x1 = Variable('x1')
    >>> x2 = Variable('x2')
    >>> target = x1 + 2 * x2
    >>> constraint = (x1 + x2 <= 10)
    '''

    name: str
    'Name of the variable.'

    @classmethod
    def sequence(cls: type[_T], prefix: str, limit: int | None = None, start: int = 1) -> Generator[_T, None, None]:
        '''
        Creates a sequence of variables with a common prefix.

        Parameters
        ----------
        prefix:
            A common prefix for generated variables.
        limit:
            Maximum number of variables generated.
        start:
            Start index of the variables.
            e.g. `prefix='y', start=3` generates from `y3`.
        
        Returns
        -------
        var_gen:
            Variable generator.
        '''
        value = start
        while limit is None or value < start + limit:
            yield cls(f'{prefix}{value}')
            value += 1
    def __init__(self, name: str):
        '''
        Initializes a new variable.

        Parameters
        ----------
        name:
            The name of the variable.
        '''
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __repr__(self):
        return self.name
    def __neg__(self):
        return (-1) * self
    def __add__(self, other: Any):
        if isinstance(other, Variable):
            return Expression({
                self: 1,
                other: 1
            })
        elif isinstance(other, Expression):
            return other.__radd__(self)
        return NotImplemented
    def __sub__(self, other: Any):
        if isinstance(other, Variable):
            return Expression({
                self: 1,
                other: -1
            })
        elif isinstance(other, Expression):
            return other.__rsub__(self)
        return NotImplemented
    def __rmul__(self, value: Any):
        if isinstance(value, Real):
            return Expression.single(self, value)
        return NotImplemented
    def __rdiv__(self, value: Any):
        return self * (1/value)
    def __le__(self, other: Any):
        return 1 * self <= other
    def __ge__(self, other: Any):
        return 1 * self >= other
    def __eq__(self, other: Any):
        return 1 * self == other

def _format_coefficient(var: Variable, coef: Real, first: bool):
    if coef == 1:
        if first:
            return f'{var}'
        else:
            return f'+ {var}'
    elif coef == -1:
        if first:
            return f'-{var}'
        else:
            return f'- {var}'
    elif first:
        return f'{coef}{var}'
    else:
        sign = '-' if coef < 0 else '+'
        return f'{sign} {coef}{var}'

class VariablePool:
    '''
    A cache pool to allow reuse of created variables.

    Example
    -------
    >>> pool = VariablePool()
    >>> x1 = pool.x1
    >>> x2 = pool.x2
    >>> x1 is pool.x1
    True
    >>> Variable("x1") is pool.x1
    False
    '''
    def __init__(self):
        '''
        Initializes a new variable pool.
        '''
        self._cache = {}
    def get(self, name: str) -> Variable:
        '''
        Gets a variable from the pool.

        Parameters
        ----------
        name:
            The name of the variable.

        Returns
        -------
        If `name` exists, return the cached variable.
        If not, creates a new one.
        '''
        if name not in self._cache:
            self._cache[name] = Variable(name)
        return self._cache[name]
    
    def __getattr__(self, name: str):
        return self.get(name)

class Expression:
    '''
    A mathematical expression with exactly one variable per item.

    Example
    -------
    >>> x1, x2, x3 = Variable.sequence('x', 2)
    >>> expr = -x1 + 4 * x2 - 2 * x3
    >>> str(expr)
    "-x1 + 4x2 - 2x3"
    >>> expr * 2
    Expression(-2x1 + 4x2 - 2x3)
    >>> expr == 10
    Constraint(-x1 + 4x2 - 2x3 == 10)
    '''
    @classmethod
    def single(cls: type[_T], var: Variable, coef: Real) -> _T:
        '''
        Returns an expression with a single item.
        Same as `coef * var`.

        Parameters
        ----------
        var:
            The variable of the expression.
        coef:
            The variable's coefficient.
        
        Returns
        -------
        expr:
            The created expression.
        '''
        return cls({ var: coef })
    def __init__(self, coefs: Mapping[Variable, Real]):
        '''
        Initializes a new expression.

        Parameters
        ----------
        coefs:
            Variable -> coefficient mapping.
        '''
        self._coefs = { var: coef for var, coef in coefs.items() if coef != 0 }
    @property
    def coefficients(self):
        '''
        Variable -> coefficient mapping.
        '''
        return self._coefs
    def display(self):
        '''
        Converts this instance to mathematical representation.

        Returns
        -------
        expr:
            Mathematical representation. Format: `{coef1}{var1} +/- {coef2}{var2} ...`.
        '''
        first = True
        result = ''
        for var, coef in self._coefs.items():
            result += ' '
            result += _format_coefficient(var, coef, first)
            if first:
                first = False
        return result.strip()
    def __repr__(self):
        return 'Expression({})'.format(self.display())
    def __str__(self):
        return self.display()
    def __neg__(self):
        coefs = { var: -coef for var, coef in self._coefs.items() }
        return Expression(coefs)
    def __add__(self, other: Any):
        if isinstance(other, Variable):
            return self + 1 * other
        if not isinstance(other, Expression):
            return NotImplemented
        coefs = self._coefs.copy()
        for var, coef in other._coefs.items():
            if var in self._coefs:
                coefs[var] += coef
            else:
                coefs[var] = coef
        return Expression(coefs)
    def __radd__(self, other: Any):
        if isinstance(other, Variable):
            return 1 * other + self
        if not isinstance(other, Expression):
            return NotImplemented
        coefs = other._coefs.copy()
        for var, coef in self._coefs.items():
            if var in self._coefs:
                coefs[var] += coef
            else:
                coefs[var] = coef
        return Expression(coefs)
    def __sub__(self, other: Any):
        return self + (-other)
    def __rsub__(self, other: Any):
        return (-self) + other
    def __mul__(self, other: Any):
        if not isinstance(other, Real):
            return NotImplemented
        coefs = { var: coef * other for var, coef in self._coefs.items() }
        return Expression(coefs)
    __rmul__ = __mul__
    def __div__(self, other: Any):
        return self * (1/other)
    def __le__(self, other: Any):
        if isinstance(other, Real):
            return Constraint(self._coefs.copy(), other, '<=')
        return NotImplemented
    def __ge__(self, other: Any):
        if isinstance(other, Real):
            return Constraint(self._coefs.copy(), other, '>=')
        return NotImplemented
    def __eq__(self, other: Any):
        if isinstance(other, Real):
            return Constraint(self._coefs.copy(), other, '==')
        return NotImplemented

OperatorType = Literal['==', '<=', '>=']
'Operator in constraint.'

class Constraint:
    '''
    A equation / inequation as a constraint of the LP problem.

    Example
    -------
    >>> x1, x2, x3 = Variable.sequence('x', 3)
    >>> cons = x1 + x2 + 3 * x3 <= 10
    >>> cons.rhs
    10
    >>> cons.coefficients
    { x1: 1, x2: 1, x3: 3 }
    >>> cons
    Constraint(x1 + x2 + 3x3 <= 10)
    >>> x4 = Variable('x4')
    >>> cons.canonicalize(x4)
    >>> cons
    Constraint(x1 + x2 + 3x3 + x4 == 10)
    '''

    rhs: Real
    'Right-hand side value.'
    operator: OperatorType
    'Operator (eq, le, ge).'

    def __init__(self, coefs: Mapping[Variable, Real], rhs: Real, operator: OperatorType):
        '''
        Initializes a new constraint.

        Parameters
        ----------
        coefs:
            Variable -> coefficient mapping.
        rhs:
            Right-hand side value.
        operator:
            Operator (eq, le, ge).
        '''
        self._coefs = { var: coef for var, coef in coefs.items() if coef != 0 }
        self.rhs = rhs
        self.operator = operator
    @property
    def coefficients(self):
        '''
        Variable -> coefficient mapping.
        '''
        return self._coefs
    @property
    def variables(self):
        '''
        All variables in this constraint.
        '''
        return set(self._coefs.keys())
    @property
    def is_canonical(self):
        '''
        Whether this constraint is an equation.
        '''
        return self.operator == '=='
    def canonicalize(self, slack_var: Variable):
        '''
        Converts this constraint into an equation.

        Parameters
        ----------
        slack_var:
            Slack variable used in canonicalization.
        
        Raises
        ------
        sapro.error.InvalidSlack:
            If `slack_var` is present in LHS expression.
        '''
        if slack_var in self._coefs:
            raise InvalidSlack('existing slack variable')
        if self.operator == '<=':
            self._coefs[slack_var] = 1
            self.operator = '=='
        elif self.operator == '>=':
            self._coefs[slack_var] = -1
            self.operator = '=='
    def display(self):
        '''
        Converts this instance to mathematical representation.

        Returns
        -------
        expr:
            Mathematical representation. Format: `{coef1}{var1} +/- {coef2}{var2} ... {operator} {rhs}`.
        '''
        return f'{Expression(self._coefs)} {self.operator} {self.rhs}'
    def __repr__(self):
        return 'Constraint({})'.format(self.display())
    def __str__(self):
        return self.display()