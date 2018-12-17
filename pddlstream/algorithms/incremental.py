import time

from pddlstream.language.statistics import load_stream_statistics, write_stream_statistics
from pddlstream.algorithms.algorithm import parse_problem, SolutionStore, add_facts, add_certified, solve_finite
from pddlstream.algorithms.instantiation import Instantiator
from pddlstream.algorithms.constraints import PlanConstraints
from pddlstream.language.conversion import revert_solution
from pddlstream.language.external import compute_instance_effort
from pddlstream.language.stream import Stream
from pddlstream.utils import INF, elapsed_time

DEFAULT_VERBOSE = True
UPDATE_STATISTICS = False

def ensure_no_fluent_streams(streams):
    for stream in streams:
        if isinstance(stream, Stream) and stream.is_fluent():
            raise NotImplementedError('Algorithm does not support fluent stream: {}'.format(stream.name))

def process_instance(instantiator, evaluations, instance, effort, verbose=False, **effort_args):
    if instance.enumerated:
        return False
    new_results, new_facts = instance.next_results(verbose=verbose)
    #if new_results and isinstance(instance, StreamInstance):
    #    evaluations.pop(evaluation_from_fact(instance.get_blocked_fact()), None)
    for result in new_results:
        for evaluation in add_certified(evaluations, result):
            instantiator.add_atom(evaluation, effort)
    for evaluation in add_facts(evaluations, new_facts, result=None): # TODO: record the instance?
        instantiator.add_atom(evaluation, effort)
    if not instance.enumerated:
        next_effort = effort + compute_instance_effort(instance, **effort_args)
        instantiator.push(instance, next_effort)
    return True

def process_function_queue(instantiator, evaluations, **kwargs):
    num_calls = 0
    while instantiator.function_queue: # not store.is_terminated()
        instance, effort = instantiator.pop_function()
        num_calls += process_instance(instantiator, evaluations, instance, effort, **kwargs)
    return num_calls

##################################################

def solve_current(problem, constraints=PlanConstraints(),
                  unit_costs=False, verbose=DEFAULT_VERBOSE, **search_kwargs):
    """
    Solves a PDDLStream problem without applying any streams
    Will fail if the problem requires stream applications
    :param problem: a PDDLStream problem
    :param constraints: PlanConstraints on the available solutions
    :param unit_costs: use unit action costs rather than numeric costs
    :param verbose: if True, this prints the result of each stream application
    :param search_kwargs: keyword args for the search subroutine
    :return: a tuple (plan, cost, evaluations) where plan is a sequence of actions
        (or None), cost is the cost of the plan, and evaluations is init but expanded
        using stream applications
    """
    evaluations, goal_expression, domain, externals = parse_problem(
        problem, constraints=constraints, unit_costs=unit_costs)
    instantiator = Instantiator(evaluations, externals)
    process_function_queue(instantiator, evaluations, verbose=verbose)
    plan, cost = solve_finite(evaluations, goal_expression, domain,
                              max_cost=constraints.max_cost, **search_kwargs)
    return revert_solution(plan, cost, evaluations)

##################################################

def solve_exhaustive(problem, constraints=PlanConstraints(),
                     unit_costs=False, max_time=300, verbose=DEFAULT_VERBOSE, **search_kwargs):
    """
    Solves a PDDLStream problem by applying all possible streams and searching once
    Requires a finite max_time when infinitely many stream instances
    :param problem: a PDDLStream problem
    :param constraints: PlanConstraints on the available solutions
    :param unit_costs: use unit action costs rather than numeric costs
    :param max_time: the maximum amount of time to apply streams
    :param verbose: if True, this prints the result of each stream application
    :param search_kwargs: keyword args for the search subroutine
    :return: a tuple (plan, cost, evaluations) where plan is a sequence of actions
        (or None), cost is the cost of the plan, and evaluations is init but expanded
        using stream applications
    """
    start_time = time.time()
    evaluations, goal_expression, domain, externals = parse_problem(
        problem, constraints=constraints, unit_costs=unit_costs)
    ensure_no_fluent_streams(externals)
    if UPDATE_STATISTICS:
        load_stream_statistics(externals)
    instantiator = Instantiator(evaluations, externals)
    while instantiator.stream_queue and (elapsed_time(start_time) < max_time):
        process_instance(instantiator, evaluations, *instantiator.pop_stream(), verbose=verbose)
    process_function_queue(instantiator, evaluations, verbose=verbose)
    plan, cost = solve_finite(evaluations, goal_expression, domain,
                              max_cost=constraints.max_cost, **search_kwargs)
    if UPDATE_STATISTICS:
        write_stream_statistics(externals, verbose)
    return revert_solution(plan, cost, evaluations)

##################################################

def process_stream_queue(instantiator, store, effort_limit, **kwargs):
    num_calls = 0
    while not store.is_terminated() and instantiator.stream_queue and (instantiator.min_effort() < effort_limit):
        num_calls += process_instance(instantiator, store.evaluations, *instantiator.pop_stream(), **kwargs)
    num_calls += process_function_queue(instantiator, store.evaluations, **kwargs)
    return num_calls

def solve_incremental(problem, constraints=PlanConstraints(),
                      unit_costs=False, success_cost=INF,
                      unit_efforts=True, max_effort=None,
                      max_iterations=INF, effort_step=1,
                      max_time=INF, verbose=DEFAULT_VERBOSE,
                      **search_kwargs):
    """
    Solves a PDDLStream problem by alternating between applying all possible streams and searching
    :param problem: a PDDLStream problem
    :param constraints: PlanConstraints on the set of legal solutions
    :param effort_step: the increase in the effort limit after each iteration
    :param max_time: the maximum amount of time to apply streams
    :param max_iterations: the maximum amount of search iterations
    :param unit_costs: use unit action costs rather than numeric costs
    :param success_cost: an exclusive (strict) upper bound on plan cost to terminate
    :param unit_efforts: use unit stream efforts rather than estimated numeric efforts
    :param max_effort: the maximum amount of effort to consider for streams
    :param verbose: if True, this prints the result of each stream application
    :param search_kwargs: keyword args for the search subroutine
    :return: a tuple (plan, cost, evaluations) where plan is a sequence of actions
        (or None), cost is the cost of the plan, and evaluations is init but expanded
        using stream applications
    """
    # success_cost = terminate_cost = decision_cost
    evaluations, goal_expression, domain, externals = parse_problem(
        problem, constraints=constraints, unit_costs=unit_costs)
    store = SolutionStore(evaluations, max_time, success_cost, verbose) # TODO: include other info here?
    ensure_no_fluent_streams(externals)
    if UPDATE_STATISTICS:
        load_stream_statistics(externals)
    num_iterations = num_calls = effort_limit = 0
    instantiator = Instantiator(evaluations, externals, unit_efforts=unit_efforts, max_effort=max_effort)
    while not store.is_terminated() and (num_iterations < max_iterations):
        num_iterations += 1
        num_calls += process_stream_queue(instantiator, store, effort_limit, unit_efforts=unit_efforts, verbose=verbose)
        print('Iteration: {} | Effort: {} | Calls: {} | Evaluations: {} | Cost: {} | Time: {:.3f}'.format(
            num_iterations, effort_limit, num_calls, len(evaluations), store.best_cost, store.elapsed_time()))
        plan, cost = solve_finite(evaluations, goal_expression, domain,
                                  max_cost=min(store.best_cost, constraints.max_cost), **search_kwargs)
        if plan is not None:
            store.add_plan(plan, cost)
        if not instantiator:
            break
        effort_limit += effort_step # TODO: option to select the next smallest effort
    if UPDATE_STATISTICS:
        write_stream_statistics(externals, verbose)
    return store.extract_solution()
