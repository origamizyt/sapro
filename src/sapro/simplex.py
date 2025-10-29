from numbers import Real
from .algebra import *
from .error import *
from .tableau import Tableau
from .utils import ftoa
from dataclasses import dataclass
from io import StringIO
from typing import Sequence, Iterator, Generator, TypedDict
import numpy as np

__all__ = ['LPStep', 'ExtraData', 'LPResult', 'FormattedConstraint', 'Simplex']

class FormattedConstraint(TypedDict):
    '''
    Represents a formatted constraint as in `Simplex.format_constraints`.
    '''
    coefficients: dict[Variable, str]
    'Variable -> coefficient mapping.'
    rhs: str
    'Right-hand side value.'
    operator: OperatorType
    'Operator (eq, le, ge).'

@dataclass
class LPStep:
    'Step of a LP problem.'

    tableau: Tableau
    'Simplex tableau.'
    enter: Variable | None
    'Variable becoming base during this step.'
    leave: Variable | None
    'Variable leaving base during this step.'

class ExtraData(TypedDict, total=False):
    'Extra data as in `LPResult.extra_data`.'

    removed_constraints: list[Constraint]
    'Removed constraints in two-phase algorithm.'
    formatted_removed_constraints: list[FormattedConstraint]
    'Formatted version of `removed_constraints`.'

@dataclass
class LPResult:
    'Result of a LP problem.'

    target_value: float
    'Minimized / Maximized target value `CT @ X`.'
    variable_values: dict[Variable, float]
    'Values of variables when the target value is reached.'
    base_variables: list[Variable]
    'Base variables selected.'
    precision: int | None
    'Precision when displaying. `None` for fraction display.'
    extra_data: ExtraData | None = None
    'Extra data e.g. `removed_constraints`.'
    @property
    def formatted_target_value(self) -> str:
        '''
        Returns a formatted version of `self.target_value`.

        Returns
        -------
        A formatted version of `self.target_value`.
        '''
        return ftoa(self.target_value, self.precision)
    @property
    def formatted_variable_values(self) -> dict[Variable, str]:
        '''
        Returns a formatted version of `self.variable_values`.

        Returns
        -------
        A dictionary like `self.variable_values`, but with every value formatted.
        '''
        return { var: ftoa(value, self.precision) for var, value in self.variable_values.items() }
    def display(self) -> str:
        '''
        Converts this object to a readable format.

        Returns
        -------
        string:
            A readable string.
            Format: `{target_value} when {var_name} = {var_value}, ...`
        '''
        result = StringIO()
        result.write(ftoa(self.target_value, self.precision))
        result.write(' when ')
        for var, value in self.variable_values.items():
            result.write(str(var))
            result.write(' = ')
            result.write(ftoa(value, self.precision))
            result.write(', ')
        return result.getvalue().rstrip(', ')
    def __str__(self):
        return self.display()

class Simplex:
    '''
    Solve LP problem using the Simplex Algorithm.

    Example
    -------
    >>> from variable import *
    >>> x = Variable.sequence('x')
    >>> x1, x2 = next(x), next(x)
    >>> problem = Simplex(
    ...     # max
    ...     x1 + 2 * x2,
    ...     # s.t.
    ...     x1 + x2 <= 4,
    ...     -2*x1 + x2 <= 1,
    ...     x1 <= 3,
    ...     maximize=True,
    ...     slack_var_generator=x
    ... )
    >>> steps = list(problem.solve()) # consumes the iterator
    >>> str(problem.result)
    "7 when x1 = 1, x2 = 3, x3 = 0, x4 = 0, x5 = 2"
    >>> problem.result.target_value
    7
    >>> problem.result.variable_values[x2]
    3
    '''

    target: Expression
    'The target expression in a LP problem.'
    constraints: list[Constraint]
    'List of constraints in a LP problem.'
    variables: list[Variable]
    'List of all variables used.'
    base_vars: list[Variable] | None
    'List of base variables.'
    maximize: bool
    'If `True`, the target will be maximized instead of minimized.'
    slack_var_generator: Iterator[Variable]
    'A iterator (has a `next` method) to generate slack variable.'
    result: LPResult | None
    'Result of the last operation.'

    def __init__(self,
                 target: Expression,
                 *constraints: Constraint,
                 vs: Sequence[Variable] | None = None,
                 bvs: Sequence[Variable] | None = None,
                 maximize: bool = False,
                 slack_var_generator: Iterator[Variable] | None = None,
                 slack_var_prefix: str = 's'):
        '''
        Initializes Simplex Algorithm.

        Parameters
        ----------
        target: Expression
            The target expression in a LP problem.
        constraints: tuple[Constraint]
            List of constraints in a LP problem.
        vs:
            List of all variables used.
            If `None`, variables will be extracted from provided constraints.
        bvs:
            List of initial base variables.
            If `None`, slack variables will be used if sufficient, otherwise
            an Exception will be raised.
            The `two_phase` method can be used to determine base variables.
        maximize:
            If `True`, the target will be maximized instead of minimized.
        slack_var_generator:
            A iterator (has a `next` method) to generate slack variable.
            If `None`, `Variable.sequence(slack_var_prefix)` will be used.
        slack_var_prefix:
            Prefix when generating slack variables.
            Ignored if `slack_var_generator` is specified.
        '''
        self.target = target
        self.constraints = list(constraints)
        self.base_vars = None if bvs is None else list(bvs)
        self.maximize = maximize
        self.slack_var_generator = slack_var_generator or Variable.sequence(slack_var_prefix)
        self.result = None
        self._num_slack_vars = 0
        if vs is None:
            variables = set()
            for c in constraints:
                variables |= c.variables
            self.variables = sorted(variables, key=lambda v: v.name)
        else:
            self.variables = list(vs)
    def display(self) -> str:
        '''
        Produces a mathematical expression of the LP problem given.

        Returns
        -------
        expr:
            A mathematical expression.

        Example
        -------
        >>> problem = Simplex(
        ...     # max
        ...     x1 + 2 * x2,
        ...     # s.t.
        ...     x1 + x2 <= 4,
        ...     -2*x1 + x2 <= 1,
        ...     x1 <= 3,
        ...     maximize=True,
        ...     slack_var_generator=x
        ... )
        >>> print(problem.display())
        max  x1 + 2x2
        s.t. x1 + x2 <= 4
             -2x1 + x2 <= 1
             x1 <= 3
        '''
        result = StringIO()
        if self.maximize:
            result.write('max  ')
        else:
            result.write('min  ')
        print(self.target, file=result)
        for i, c in enumerate(self.constraints):
            if i == 0:
                print('s.t.', c, file=result)
            else:
                print('    ', c, file=result)
        return result.getvalue().rstrip()
    def __str__(self):
        return self.display()
    def __repr__(self):
        return 'Simplex({}, {})'.format(len(self.variables), len(self.constraints))
    def _select_entering_var(self, sigma: np.ndarray, maximize: bool) -> int | None:
        '''
        Selects a variable to become base.

        Parameters
        ---------
        sigma:
            The checksum array.
        arbitrary:
            If `True`, just pick a non-zero element regardless of `self.maximize`.
            Used in method `two_phase`.
        
        Returns
        -------
        index:
            Index of the selected variable. `None` if none available.
        '''
        if maximize:
            candidates = (sigma > 0).nonzero()[0]
        else:
            candidates = (sigma < 0).nonzero()[0]
        return candidates[0] if candidates.shape[0] > 0 else None
    def _select_leaving_var(self, rhs: np.ndarray) -> int | None:
        '''
        Selects a variable to leave base.

        Parameters
        ----------
        rhs:
            The right-hand side array.
        
        Returns
        -------
        index:
            Index of the selected variable. `None` if none available.
        '''
        candidates = (rhs < 0).nonzero()[0]
        return candidates[0] if candidates.shape[0] > 0 else None
    def canonicalize(self) -> int:
        '''
        Canonicalizes all constraints using slack variables from `self.slack_var_generator`.

        Returns
        -------
        num_slack_vars:
            Number of slack variables used.
        '''
        for c in self.constraints:
            if not c.is_canonical:
                self._num_slack_vars += 1
                v = next(self.slack_var_generator)
                c.canonicalize(v)
                self.variables.append(v)
        return self._num_slack_vars
    def format_constraints(self, precision: int | None = None) -> list[FormattedConstraint]:
        '''
        Formats the coefficients in constraints.

        Parameters
        ----------
        precision:
            Number of digits after the floating point.
            If `None` or negative, fractions will be used.
        
        Returns
        -------
        formatted_constraints:
            The formatted constraints.
        '''
        return [
            {
                "coefficients": { var: ftoa(coef, precision) for var, coef in c.coefficients.items() },
                "rhs": ftoa(c.rhs, precision),
                "operator": c.operator
            }
            for c in self.constraints
        ]
    def _prepare(self, need_base_vars: bool):
        '''
        Prepares for simplex iterations.
        Calls `canonicalize` and checks the number of base variables.

        Raises
        ------
        sapro.error.InvalidBase:
            Raised if one of the following occurs:
            - No bases variables set, and the number of slack variables doesn't match the number of constraints.
            - The number of base variables doesn't match the number of constraints.
        '''
        self.canonicalize()
        # check variables
        M = len(self.constraints)
        if M <= 0:
            raise Unsolvable('constraints is empty')
        if not need_base_vars:
            return
        if self.base_vars is None:
            if self._num_slack_vars == M:
                self.base_vars = self.variables[-M:]
            else:
                raise InvalidBase('cannot determine base variables')
        elif len(self.base_vars) != len(self.constraints):
            raise InvalidBase('number of base variables doesn\'t match number of constraints')
    def _init_matrices(self, 
                       constraints: Sequence[Constraint], 
                       variables: Sequence[Variable], 
                       target: Expression, 
                       base_vars: Sequence[Variable]) -> tuple[np.ndarray, ...]:
        '''
        Initialize `data`, `sigma`, `rhs` and `z` values.

        Parameters
        ----------
        constraints:
            A sequence of constraints.
        variables:
            A sequence of all variables.
        target:
            The target expression.
        base_vars:
            A sequence of base variables.
        
        Returns
        -------
        data, sigma, rhs, z:
            Initialized values.    
        '''

        # check variables
        M = len(constraints)
        N = len(variables)

        # initialize Ax = b, z = cTx matrices
        A = np.zeros((M, N))
        b = np.zeros(M)
        cT = np.zeros(N)
        for i, c in enumerate(constraints):
            for var, coef in c.coefficients.items():
                A[i][variables.index(var)] = coef
            b[i] = c.rhs
        for var, coef in target.coefficients.items():
            cT[variables.index(var)] = coef
        
        # initialize AB, AN, cTB, cTN matrices
        base_indices = [variables.index(base_var) for base_var in base_vars]
        non_base_indices = [i for i in range(N) if i not in base_indices]
        AB = A[:, base_indices]
        AN = A[:, non_base_indices]
        cTB = cT[base_indices]
        cTN = cT[non_base_indices]

        # initialize data, sigma, rhs and z
        ABinv = np.linalg.inv(AB)
        ABinvAN = ABinv @ AN
        data = np.zeros((M, N))
        data[:, base_indices] = np.eye(M)
        data[:, non_base_indices] = ABinvAN
        sigma = np.zeros(N)
        sigma[non_base_indices] = cTN - cTB @ ABinvAN
        rhs = ABinv @ b
        z = -cTB @ rhs

        return data, sigma, rhs, z
    def set_base_vars(self, base_vars: Sequence[Variable] | None):
        '''
        Sets the base variables before solving.

        Parameters
        ----------
        base_vars:
            The base variable sequence.
        '''
        self.base_vars = base_vars
    def two_phase(self, 
                  yield_initial_tableau: bool = False, 
                  precision: int | None = None, 
                  artificial_var_prefix: str = 'u') -> Generator[LPStep, None, None]:
        '''
        Calculates BFS (Basic Feasible Solution) using the Two-phase Algorithm.
        Sets `self.result` to the BFS.

        Parameters
        ----------
        yield_initial_tableau:
            If `True`, the initial tableau (before any variable enters) will be yielded.
        precision:
            The precision of the result when displayed.
            If `None`, display using closest fraction.
        artificial_var_prefix:
            Prefix when generating artificial variables.

        Raises
        ------
        sapro.error.InvalidBase:
            Raised if one of the following occurs:
            - No bases variables set, and the number of slack variables doesn't match the number of constraints.
            - The number of base variables doesn't match the number of constraints.
        sapro.error.BaseAlreadySet:
            If base variables are already set.
        sapro.error.Cycle:
            If cycles appear while iterating.
        
        Returns
        -------
        tableau_gen:
            A generator, which yields at each algorithm step with the simplex table.
        '''
        self._prepare(False)
        if self.base_vars is not None:
            raise BaseAlreadySet('base variables already set')
        artificial_vars = list(Variable.sequence(artificial_var_prefix, len(self.constraints)))
        base_vars = artificial_vars.copy()
        variables = self.variables.copy()
        variables.extend(artificial_vars)

        M = len(self.constraints)

        constraints = []
        for c, avar in zip(self.constraints, artificial_vars):
            coefs = c.coefficients.copy()
            coefs[avar] = 1
            constraints.append(Constraint(coefs, c.rhs, c.operator))

        data, sigma, rhs, z = self._init_matrices(
            constraints,
            variables,
            Expression(dict.fromkeys(artificial_vars, 1)),
            base_vars
        )

        if yield_initial_tableau:
            yield LPStep(
                tableau=Tableau(
                    data.copy(),
                    sigma.copy(),
                    rhs.copy(),
                    z,
                    variables.copy(),
                    base_vars.copy(),
                    precision=precision
                ),
                enter=None,
                leave=None
            )
        
        base_var_memo = set()
        base_var_memo.add(frozenset(base_vars))

        # algorithm step
        while True:
            enter_index = self._select_entering_var(sigma[:-M], False)
            if enter_index is None:
                break
            with np.errstate(divide='ignore'):
                ratios = np.where(
                    data[:, enter_index] <= 0,
                    np.inf,
                    rhs / data[:, enter_index]
                )
            leave_index = np.argmin(ratios)
            leave_var = base_vars[leave_index]
            rhs[leave_index] /= data[leave_index][enter_index]
            data[leave_index] /= data[leave_index][enter_index]
            for i in range(M):
                if i == leave_index:
                    continue
                if data[i][enter_index] == 0:
                    continue
                rhs[i] -= rhs[leave_index] * data[i][enter_index]
                data[i] -= data[leave_index] * data[i][enter_index]
            ratio = sigma[enter_index] / data[leave_index][enter_index]
            sigma -= data[leave_index] * ratio
            z -= rhs[leave_index] * ratio
            base_vars[leave_index] = variables[enter_index]

            yield LPStep(
                tableau=Tableau(
                    data.copy(),
                    sigma.copy(),
                    rhs.copy(),
                    z,
                    variables.copy(),
                    base_vars.copy(),
                    precision=precision
                ),
                enter=variables[enter_index],
                leave=leave_var
            )

            base_var_set = frozenset(base_vars)
            if base_var_set in base_var_memo:
                raise Boundless("encountered cycle in simplex")
            base_var_memo.add(base_var_set)
        
        if not np.isclose(z, 0) or (rhs < 0).any():
            raise Unsolvable('cannot find feasible solution')

        removed_constraints = []
        for avar in artificial_vars:
            if avar in base_vars:
                # useless constraint
                removed_constraints.append(
                    self.constraints.pop(base_vars.index(avar))
                )
                base_vars.remove(avar)
        self.set_base_vars(base_vars)

        cT = np.zeros(len(self.variables))
        for var, coef in self.target.coefficients.items():
            cT[self.variables.index(var)] = coef
        x = np.zeros(len(self.variables))
        for var, coef in zip(base_vars, rhs):
            x[self.variables.index(var)] = coef
        var_values = dict.fromkeys(variables, 0)
        var_values.update(zip(base_vars, rhs))
        # do not include artificial variables here.
        for avar in artificial_vars:
            del var_values[avar]
        self.result = LPResult(
            target_value=cT @ x,
            variable_values=var_values,
            precision=precision,
            base_variables=base_vars.copy(),
            extra_data={
                'removed_constraints': removed_constraints,
                'formatted_removed_constraints': [
                    {
                        "coefficients": { var: ftoa(coef, precision) for var, coef in c.coefficients.items() },
                        "rhs": ftoa(c.rhs, precision),
                        "operator": c.operator
                    }
                    for c in removed_constraints
                ]
            }
        )

    def solve(self, 
              yield_initial_tableau: bool = False, 
              precision: int | None = None) -> Generator[LPStep, None, None]:
        '''
        Calculates the best solution using the Simplex Algorithm.
        Sets `self.result` to the solution.

        Parameters
        ----------
        yield_initial_tableau:
            If `True`, the initial tableau (before any variable enters) will be yielded.
        precision:
            The precision of the result when displayed.
            If `None`, display using closest fraction.

        Raises
        ------
        sapro.error.InvalidBase:
            Raised if one of the following occurs:
            - No bases variables set, and the number of slack variables doesn't match the number of constraints.
            - The number of base variables doesn't match the number of constraints.
        sapro.error.Cycle:
            If cycles appear while iterating.
        
        Returns
        -------
        tableau_gen:
            A generator, which yields at each algorithm step with the simplex table.
        '''
        self._prepare(True)

        # check variables
        M = len(self.constraints)

        # initialize matrices
        data, sigma, rhs, z = self._init_matrices(
            self.constraints,
            self.variables,
            self.target,
            self.base_vars
        )

        if yield_initial_tableau:
            yield LPStep(
                tableau=Tableau(
                    data.copy(),
                    sigma.copy(),
                    rhs.copy(),
                    z,
                    self.variables.copy(),
                    self.base_vars.copy(),
                    precision=precision
                ),
                enter=None,
                leave=None
            )
        
        base_var_memo = set()
        base_var_memo.add(frozenset(self.base_vars))

        # algorithm step
        while True:
            enter_index = self._select_entering_var(sigma, self.maximize)
            if enter_index is None:
                break
            with np.errstate(divide='ignore'):
                ratios = np.where(
                    data[:, enter_index] <= 0,
                    np.inf,
                    rhs / data[:, enter_index]
                )
            leave_index = np.argmin(ratios)
            leave_var = self.base_vars[leave_index]
            rhs[leave_index] /= data[leave_index][enter_index]
            data[leave_index] /= data[leave_index][enter_index]
            for i in range(M):
                if i == leave_index:
                    continue
                if data[i][enter_index] == 0:
                    continue
                rhs[i] -= rhs[leave_index] * data[i][enter_index]
                data[i] -= data[leave_index] * data[i][enter_index]
            ratio = sigma[enter_index] / data[leave_index][enter_index]
            sigma -= data[leave_index] * ratio
            z -= rhs[leave_index] * ratio
            self.base_vars[leave_index] = self.variables[enter_index]
            
            yield LPStep(
                tableau=Tableau(
                    data.copy(),
                    sigma.copy(),
                    rhs.copy(),
                    z,
                    self.variables.copy(),
                    self.base_vars.copy(),
                    precision=precision
                ),
                enter=self.variables[enter_index],
                leave=leave_var
            )

            base_var_set = frozenset(self.base_vars)
            if base_var_set in base_var_memo:
                raise Boundless("encountered cycle in simplex")
            base_var_memo.add(base_var_set)

        base_var_memo.clear()
        while True:
            leave_index = self._select_leaving_var(rhs)
            if leave_index is None:
                break
            leave_var = self.base_vars[leave_index]
            enterable = data[leave_index] < 0
            if not enterable.any():
                raise Unsolvable(f'cannot make "{leave_var}" leave base')
            with np.errstate(divide='ignore'):
                ratios = np.where(
                    enterable,
                    sigma / data[leave_index],
                    np.inf,
                )
            enter_index = np.argmin(np.abs(ratios))
            rhs[leave_index] /= data[leave_index][enter_index]
            data[leave_index] /= data[leave_index][enter_index]
            for i in range(M):
                if i == leave_index:
                    continue
                if data[i][enter_index] == 0:
                    continue
                rhs[i] -= rhs[leave_index] * data[i][enter_index]
                data[i] -= data[leave_index] * data[i][enter_index]
            ratio = sigma[enter_index] / data[leave_index][enter_index]
            sigma -= data[leave_index] * ratio
            z -= rhs[leave_index] * ratio
            self.base_vars[leave_index] = self.variables[enter_index]
            
            yield LPStep(
                tableau=Tableau(
                    data.copy(),
                    sigma.copy(),
                    rhs.copy(),
                    z,
                    self.variables.copy(),
                    self.base_vars.copy(),
                    precision=precision
                ),
                enter=self.variables[enter_index],
                leave=leave_var
            )

            base_var_set = frozenset(self.base_vars)
            if base_var_set in base_var_memo:
                raise Boundless("encountered cycle in simplex")
            base_var_memo.add(base_var_set)
        
        var_values = dict.fromkeys(self.variables, 0)
        var_values.update(zip(self.base_vars, rhs))
        self.result = LPResult(
            target_value=-z,
            variable_values=var_values,
            base_variables=self.base_vars.copy(),
            precision=precision,
        )