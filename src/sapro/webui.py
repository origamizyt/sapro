from .algebra import *
from .error import LPError
from .simplex import *
from .tableau import Tableau
from .utils import *
from wsgiref.simple_server import make_server
from wsgiref.util import FileWrapper
from http import HTTPStatus
import os

def _variable_encoder(value: Variable, *_):
    return value.name

def _expression_encoder(value: Expression, *_):
    mapping = {}
    for var, coef in value.coefficients.items():
        mapping[var.name] = coef
    return mapping

def _constraint_encoder(value: Constraint, *_):
    mapping = _expression_encoder(Expression(value.coefficients))
    return {
        'coefficients': mapping,
        'rhs': value.rhs,
        'operator': value.operator
    }

def _lperror_encoder(value: LPError, *_):
    return {
        'message': str(value),
        'type': type(value).__name__
    }

def _tableau_encoder(value: Tableau, *_):
    return value.to_frame()

def _make_expression_decoder(pool: VariablePool):
    def decoder(value, *_):
        mapping = {}
        for name, coef in value.items():
            mapping[pool.get(name)] = coef
        return Expression(mapping)
    return decoder

def _make_constraint_decoder(pool: Constraint):
    def decoder(value, *_):
        mapping = _make_expression_decoder(pool)(value['coefficients']).coefficients
        return Constraint(
            mapping,
            value['rhs'],
            value['operator']
        )
    return decoder

@Parcelable
class SolveParams:
    '''
    Parameters sent from client in POST request.
    '''
    target: Expression
    constraints: list[Constraint]
    maximize: bool
    base_variables: list[Variable] | None
    needs_two_phase: bool
    slack_variable_start: str
    artificial_variable_prefix: str | None
    show_canonicalization: bool
    yield_initial_tableau: bool
    precision: int | None

@Parcelable
class SolveResult:
    '''
    Result sent to client in POST request.
    '''
    ok: bool
    error: LPError | None
    slack_variable_count: int | None
    canonical_constraints: list[Constraint] | None
    two_phase_steps: list[LPStep] | None
    solve_steps: list[LPStep] | None
    feasible_result: LPResult | None
    best_result: LPResult | None

def _http_status_line(status: HTTPStatus):
    return '{} {}'.format(int(status), status.phrase)

def application(environ, start_response):
    '''
    WSGI application of the server.
    '''
    if environ['PATH_INFO'] != '/':
        start_response(_http_status_line(HTTPStatus.NOT_FOUND), [])
        return []
    if environ['REQUEST_METHOD'] == 'GET':
        html_path = os.path.join(os.path.dirname(__file__), 'index.html')
        start_response(_http_status_line(HTTPStatus.OK), [
            ('Content-Type', 'text/html')
        ])
        return FileWrapper(open(html_path, 'rb'))
    elif environ['REQUEST_METHOD'] != 'POST':
        start_response(_http_status_line(HTTPStatus.METHOD_NOT_ALLOWED), [])
        return []
    
    pool = VariablePool()
    temporary_decoders = {
        Variable: lambda name, *_: pool.get(name),
        Expression: _make_expression_decoder(pool),
        Constraint: _make_constraint_decoder(pool)
    }
    body_size = int(environ['CONTENT_LENGTH'])
    body = environ['wsgi.input'].read(body_size)

    try:
        params = JSONConverter.decode_json(body, SolveParams, temporary_decoders)
    except TypeError as e:
        start_response(_http_status_line(HTTPStatus.BAD_REQUEST), [(
            'Content-Type', 'text/plain'
        )])
        return [str(e).encode()]
    
    prefix, start = parse_variable(params.slack_variable_start, 's')
    slack_var_generator = Variable.sequence(prefix, start=start)

    problem = Simplex(
        params.target,
        *params.constraints,
        maximize=params.maximize,
        slack_var_generator=slack_var_generator
    )

    result = SolveResult(ok=True)
    try:
        if params.show_canonicalization:
            result.slack_variable_count = problem.canonicalize()
            result.canonical_constraints = problem.constraints.copy()
        if params.base_variables:
            problem.set_base_vars(params.base_variables)
        if params.needs_two_phase:
            result.two_phase_steps = []
            for step in problem.two_phase(
                params.yield_initial_tableau,
                params.precision,
                params.artificial_variable_prefix or 'u'
            ):
                result.two_phase_steps.append(step)
            result.feasible_result = problem.result
        result.solve_steps = []
        for step in problem.solve(
            params.yield_initial_tableau,
            params.precision
        ):
            result.solve_steps.append(step)
        result.best_result = problem.result
    except LPError as e:
        result.ok = False
        result.error = e
    
    start_response(_http_status_line(HTTPStatus.OK), [
        ('Content-Type', 'application/json')
    ])
    return [JSONConverter.encode_json(result, SolveResult).encode()]

def run_app(host: str = '0.0.0.0', port: int = 80):
    '''
    Runs the server on specified address.

    Parameters
    ----------
    host:
        The host to serve on.
    port:
        The port to serve on.
    '''
    JSONConverter.register(Variable, encoder=_variable_encoder)
    JSONConverter.register(Expression, encoder=_expression_encoder)
    JSONConverter.register(Constraint, encoder=_constraint_encoder)
    JSONConverter.register(Tableau, encoder=_tableau_encoder)
    JSONConverter.register(LPError, encoder=_lperror_encoder, register_subclasses=True)
    Parcelable.mark_dataclass(LPStep)
    Parcelable.mark_dataclass(LPResult, 
                              formatted_target_value=str,
                              formatted_variable_values=dict[Variable, str])
    Parcelable.mark_typeddict(ExtraData)
    server = make_server(host, port, application)
    server.serve_forever()