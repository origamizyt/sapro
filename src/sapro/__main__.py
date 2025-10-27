from .algebra import *
from .simplex import Simplex
from .utils import parse_variable
from .webui import run_app
from fractions import Fraction
from threading import Thread
import argparse, time, webbrowser

def parse_constraint(s: str, pool: VariablePool):
    s = s.strip()
    for op in OperatorType.__args__:
        if op in s:
            parts = s.split(op)
            coefs = parse_expression(parts[0], pool).coefficients
            rhs = Fraction(parts[1])
            return Constraint(coefs, rhs, op)
    raise ValueError(f'cannot find operator in constraint ${s!r}')

def parse_expression(s: str, pool: VariablePool):
    s = s.strip()
    d = {}
    for outer_index, add_part in enumerate(s.split('+')):
        add_part = add_part.strip()
        for index, part in enumerate(add_part.split('-')):
            part = part.strip()
            if index == 0:
                if outer_index == 0 and not part:
                    continue # leading +/-, fine.
                v, coef = parse_expr_item(part, pool)
                if v in d:
                    d[v] += coef
                else:
                    d[v] = coef
            else:
                v, coef = parse_expr_item(part, pool)
                if v in d:
                    d[v] -= coef
                else:
                    d[v] = -coef
    return Expression(d)

def parse_expr_item(s: str, pool: VariablePool):
    s = s.strip()
    parts = s.split('*')
    if len(parts) == 1:
        return pool.get(parts[0].strip()), 1
    elif len(parts) == 2:
        return pool.get(parts[1].strip()), Fraction(parts[0].strip())
    raise ValueError(f'invalid entry {s!r}')

def command_webui(parser):
    'Launches http server for Sapro.'
    parser.add_argument('-H', '--host', default='0.0.0.0', help='Host to serve on.')
    parser.add_argument('-p', '--port', type=int, default=5678, help='Port to serve on.')
    parser.add_argument('--open', action='store_true', help='Open UI in browser.')
    args = yield
    try:
        print('Serving on {}:{}...'.format(args.host, args.port))
        print('Press Ctrl+C to quit.')
        t = Thread(target=run_app, args=(args.host, args.port), daemon=True)
        t.start()
        if args.open:
            webbrowser.open(f'http://localhost:{args.port}', 2)
        while True:
            time.sleep(0xFF)
    except KeyboardInterrupt: pass

def command_solve(parser):
    'Solves a LP problem using simplex.'
    parser.add_argument('target', help='The target of the LP problem.')
    parser.add_argument('-c', '--constraints', action='append', help='The constraints of the LP problem.')
    parser.add_argument('--max', action='store_true', help='Maximize target instead of minimizing it.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Display simplex tableaus.')
    parser.add_argument('--include-first', action='store_true', help='Whether to display the first tableau or not.')
    parser.add_argument('-b', '--bases', nargs='*', help='The bases of the LP problem. To use the Two-Phase algorithm, specify "two-phase".')
    parser.add_argument('--slack', required=True, help='Slack variable naming will start from this argument.')
    parser.add_argument('-p', '--precision', type=int, default=-1, help='The precision of the output. Default: fraction.')
    args = yield
    pool = VariablePool()
    target = parse_expression(args.target, pool)
    constraints = [parse_constraint(c, pool) for c in args.constraints]
    if args.bases:
        if args.bases[0] == 'two-phase':
            needs_two_phase = True
            bases = None
        else:
            needs_two_phase = False
            bases = [Variable(x) for x in args.bases]
    else:
        needs_two_phase = False
        bases = None
    slack_prefix, slack_index = parse_variable(args.slack, None)
    slack_gen = Variable.sequence(slack_prefix, start=slack_index)
    problem = Simplex(
        target,
        *constraints,
        bvs=bases,
        maximize=args.max,
        slack_var_generator=slack_gen
    )
    if needs_two_phase:
        for i, step in enumerate(problem.two_phase(
            yield_initial_tableau=args.include_first,
            precision=args.precision)):
            if args.verbose:
                print(f'Step {i+1}: ({step.enter} enters, {step.leave} leaves)')
                print(step.tableau)

    for i, step in enumerate(problem.solve(
        yield_initial_tableau=args.include_first,
        precision=args.precision)):
        if args.verbose:
            print(f'Step {i+1}: ({step.enter} enters, {step.leave} leaves)')
            print(step.tableau)
    
    print('z =', problem.result)

COMMANDS = {
    'solve': command_solve,
    'webui': command_webui
}

def main():
    parser = argparse.ArgumentParser(description='Command line tools for Sapro.')
    subparsers = parser.add_subparsers(required=True)
    for name, func in COMMANDS.items():
        subparser = subparsers.add_parser(name, help=func.__doc__)
        g = func(subparser)
        subparser.set_defaults(handler=g)
        next(g)
    
    args = parser.parse_args()
    try:
        args.handler.send(args)
    except StopIteration: pass

if __name__ == '__main__':
    main()