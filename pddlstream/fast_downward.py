from __future__ import print_function

import os
import sys
from time import time
from collections import namedtuple

from pddlstream.utils import read, write, ensure_dir, safe_rm_dir, INF
import re

FD_BIN = os.path.join(os.environ['FD_PATH'], 'bin')
TRANSLATE_PATH = os.path.join(FD_BIN, 'translate')

DOMAIN_INPUT = 'domain.pddl'
PROBLEM_INPUT = 'problem.pddl'
TRANSLATE_FLAGS = [] # '--negative-axioms'
sys.argv = sys.argv[:1] + TRANSLATE_FLAGS + [DOMAIN_INPUT, PROBLEM_INPUT]
sys.path.append(TRANSLATE_PATH)

import translate
import pddl
from pddl_parser.lisp_parser import parse_nested_list, tokenize
from pddl_parser.parsing_functions import parse_domain_pddl, parse_task_pddl, check_for_duplicates
import normalize
import pddl_parser

TEMP_DIR = 'temp/'
TRANSLATE_OUTPUT = 'output.sas'
SEARCH_OUTPUT = 'sas_plan'
SEARCH_COMMAND = 'downward --internal-plan-file %s %s < %s'

# TODO: be careful when doing costs. Might not be admissible if use plus one for heuristic
# TODO: use goal_serialization / hierarchy on the inside loop of these algorithms

SEARCH_OPTIONS = {
    # Optimal
    'dijkstra': '--heuristic "h=blind(transform=adapt_costs(cost_type=NORMAL))" '
                '--search "astar(h,cost_type=NORMAL,max_time=%s,bound=%s)"',
    'max-astar': '--heuristic "h=hmax(transform=adapt_costs(cost_type=NORMAL))"'
                 ' --search "astar(h,cost_type=NORMAL,max_time=%s,bound=%s)"',

    # Suboptimal
    'ff-astar': '--heuristic "h=ff(transform=adapt_costs(cost_type=NORMAL))" '
                '--search "astar(h,cost_type=NORMAL,max_time=%s,bound=%s)"',
    'ff-wastar1': '--heuristic "h=ff(transform=adapt_costs(cost_type=NORMAL))" '
                  '--search "lazy_wastar([h],preferred=[h],reopen_closed=true,boost=100,w=1,'
                  'preferred_successors_first=true,cost_type=NORMAL,max_time=%s,bound=%s)"',
    'ff-wastar3': '--heuristic "h=ff(transform=adapt_costs(cost_type=PLUSONE))" '
                  '--search "lazy_wastar([h],preferred=[h],reopen_closed=false,boost=100,w=3,'
                  'preferred_successors_first=true,cost_type=PLUSONE,max_time=%s,bound=%s)"',
    'ff-wastar5': '--heuristic "h=ff(transform=adapt_costs(cost_type=PLUSONE))" '
                  '--search "lazy_wastar([h],preferred=[h],reopen_closed=false,boost=100,w=5,'
                  'preferred_successors_first=true,cost_type=PLUSONE,max_time=%s,bound=%s)"',

    'cea-wastar1': '--heuristic "h=cea(transform=adapt_costs(cost_type=PLUSONE))" '
                   '--search "lazy_wastar([h],preferred=[h],reopen_closed=false,boost=1000,w=1,'
                   'preferred_successors_first=true,cost_type=PLUSONE,max_time=%s,bound=%s)"',
    'cea-wastar3': '--heuristic "h=cea(transform=adapt_costs(cost_type=PLUSONE))" '
                   '--search "lazy_wastar([h],preferred=[h],reopen_closed=false,boost=1000,w=3,'
                   'preferred_successors_first=true,cost_type=PLUSONE,max_time=%s,bound=%s)"',
    'cea-wastar5': '--heuristic "h=cea(transform=adapt_costs(cost_type=PLUSONE))" '
                   '--search "lazy_wastar([h],preferred=[h],reopen_closed=false,boost=1000,w=5,'
                   'preferred_successors_first=true,cost_type=PLUSONE,max_time=%s,bound=%s)"',

    'ff-eager': '--heuristic "hff=ff(transform=adapt_costs(cost_type=PLUSONE))" '
                '--search "eager_greedy([hff],max_time=%s,bound=%s)"',
    'ff-eager-pref': '--heuristic "hff=ff(transform=adapt_costs(cost_type=PLUSONE))" '
                     '--search "eager_greedy([hff],preferred=[hff],max_time=%s,bound=%s)"',
    'ff-lazy': '--heuristic "hff=ff(transform=adapt_costs(cost_type=PLUSONE))" '
               '--search "lazy_greedy([hff],preferred=[hff],max_time=%s,bound=%s)"',
}


##################################################

from pddlstream.conversion import objects_from_evaluations

def get_problem(init_evaluations, goal_expression, domain, metric=False):
    objects = objects_from_evaluations(init_evaluations)
    s = '(define (problem {})\n' \
           '\t(:domain {})\n' \
           '\t(:objects {})\n' \
           '\t(:init {})\n' \
           '\t(:goal {})'.format(problem_name, domain_name,
                                 pddl_from_objects(objects),
                                 pddl_from_evaluations(init_evaluations),
                                 pddl_from_expression(goal_expression))
    #objective = None # minimizes length
    if objective is not None:
        s += '\n\t(:metric minimize {})'.format(
            pddl_from_expression(objective))
    return s + ')\n'

def parse_lisp(lisp):
    return parse_nested_list(lisp.split())

Domain = namedtuple('Domain', ['name', 'requirements', 'types', 'type_dict', 'constants',
                               'predicates', 'predicate_dict', 'functions', 'actions', 'axioms'])

def parse_domain(domain_pddl):
    return Domain(*parse_domain_pddl(parse_lisp(domain_pddl)))

Problem = namedtuple('Problem', ['task_name', 'task_domain_name', 'task_requirements', 'objects', 'init',
                               'goal', 'use_metric'])

def parse_problem(domain, problem_pddl):
    return Problem(*parse_task_pddl(parse_lisp(problem_pddl), domain.type_dict, domain.predicate_dict))

def task_from_domain_problem(domain, problem):
    domain_name, domain_requirements, types, type_dict, constants, predicates, predicate_dict, functions, actions, axioms \
                 = domain
    task_name, task_domain_name, task_requirements, objects, init, goal, use_metric = problem

    assert domain_name == task_domain_name
    requirements = pddl.Requirements(sorted(set(
                domain_requirements.requirements +
                task_requirements.requirements)))
    objects = constants + objects
    check_for_duplicates(
        [o.name for o in objects],
        errmsg="error: duplicate object %r",
        finalmsg="please check :constants and :objects definitions")
    init += [pddl.Atom("=", (obj.name, obj.name)) for obj in objects]

    return pddl.Task(
        domain_name, task_name, requirements, types, objects,
        predicates, functions, init, goal, actions, axioms, use_metric)

def translate_task(task, temp_dir):
    normalize.normalize(task)
    sas_task = translate.pddl_to_sas(task)
    translate.dump_statistics(sas_task)
    with open(os.path.join(temp_dir, TRANSLATE_OUTPUT), "w") as output_file:
        sas_task.output(output_file)
    return sas_task

def translate_pddl(domain_path, problem_path):
    #pddl_parser.parse_pddl_file('domain', domain_path)
    #pddl_parser.parse_pddl_file('task', problem_path)
    task = pddl_parser.open(
        domain_filename=domain_path, task_filename=problem_path)
    normalize.normalize(task)
    return task

def run_translate2(domain_pddl, problem_pddl, temp_dir):
    domain = parse_domain(domain_pddl)
    problem = parse_problem(domain, problem_pddl)
    task = task_from_domain_problem(domain, problem)
    translate_task(task, temp_dir)

def run_translate(temp_dir, verbose):
    t0 = time()
    old_cwd = os.getcwd()
    tmp_cwd = os.path.join(old_cwd, temp_dir)
    if verbose:
        print('\nTranslate command: import translate; translate.main()')
        os.chdir(tmp_cwd)
        translate.main()
        os.chdir(old_cwd)
        print('Translate runtime:', time() - t0)
        return

    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        os.chdir(tmp_cwd)
        try:
            translate.main()
        finally:
            sys.stdout = old_stdout
            os.chdir(old_cwd)


def run_search(temp_dir, planner, max_time, max_cost, verbose):
    if max_time == INF:
        max_time = 'infinity'
    elif isinstance(max_time, float):
        max_time = int(max_time)
    if max_cost == INF:
        max_cost = 'infinity'
    elif isinstance(max_cost, float):
        max_cost = int(max_cost)

    t0 = time()
    search = os.path.join(FD_BIN, SEARCH_COMMAND)
    planner_config = SEARCH_OPTIONS[planner] % (max_time, max_cost)
    command = search % (temp_dir + SEARCH_OUTPUT, planner_config, temp_dir + TRANSLATE_OUTPUT)
    if verbose:
        print('\nSearch command:', command)
    p = os.popen(command)  # NOTE - cannot pipe input easily with subprocess
    output = p.read()
    if verbose:
        print(output[:-1])
        print('Search runtime:', time() - t0)
    if not os.path.exists(temp_dir + SEARCH_OUTPUT):
        return None
    return read(temp_dir + SEARCH_OUTPUT)


##################################################

def parse_solution(solution):
    #action_regex = r'\((\w+(\s+\w+)\)' # TODO: regex
    cost = INF
    if solution is None:
        return None, cost
    cost_regex = r'cost\s*=\s*(\d+)'
    matches = re.findall(cost_regex, solution)
    if matches:
        cost = int(matches[0])
    lines = solution.split('\n')[:-2]  # Last line is newline, second to last is cost
    plan = []
    for line in lines:
        entries = line.strip('( )').split(' ')
        plan.append((entries[0], list(entries[1:])))
    return plan, cost

def write_pddl(domain_pddl=None, problem_pddl=None, temp_dir=TEMP_DIR):
    safe_rm_dir(temp_dir)
    ensure_dir(temp_dir)
    domain_path = os.path.join(temp_dir, DOMAIN_INPUT)
    if domain_pddl is not None:
        write(domain_path, domain_pddl)
    problem_path = os.path.join(temp_dir, PROBLEM_INPUT)
    if problem_pddl is not None:
        write(problem_path, problem_pddl)
    return domain_path, problem_path

def run_fast_downward(domain_pddl, problem_pddl, planner='max-astar',
                      max_time=INF, max_cost=INF, verbose=False, clean=False, temp_dir=TEMP_DIR):
    start_time = time()
    domain_path, problem_path = write_pddl(domain_pddl, problem_pddl, temp_dir)
    #run_translate(temp_dir, verbose)
    run_translate2(domain_pddl, problem_pddl, temp_dir)
    solution = run_search(temp_dir, planner, max_time, max_cost, verbose)
    if clean:
        safe_rm_dir(temp_dir)
    if verbose:
        print('Total runtime:', time() - start_time)
    return parse_solution(solution)
