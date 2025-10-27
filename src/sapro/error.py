class LPError(ArithmeticError):
    'Base class for linear programming errors.'

class InvalidBase(LPError):
    'Invalid base variables are provided.'

class InvalidSlack(LPError):
    'Invalid slack variables are provided.'

class BaseAlreadySet(LPError):
    'In `two_phases`, raised of previous calls to `set_base_vars` have been made.'

class Unsolvable(LPError):
    'A LP problem has no solution.'
