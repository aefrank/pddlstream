#!/usr/bin/env python

import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Callable

import argparse
import os
import functools as ft
from igibson.external.pybullet_tools.utils import stable_z
import numpy as np

### PDDLStream-specific imports
from pddlstream.algorithms.meta import ALGORITHMS, create_parser, solve
from pddlstream.utils \
    import INF, Profiler, negate_test, read, str_from_object
from pddlstream.language.constants \
    import PDDLProblem, print_solution
from pddlstream.language.generator \
    import from_fn, from_gen_fn, from_test

# Custom imports
from examples.fetch.from_kuka_tamp.fetch_primitives import LockRenderer, MyiGibsonSemanticInterface
from examples.fetch.from_kuka_tamp.utils.helpers import fmt_fluents
from examples.fetch.from_kuka_tamp.utils.stream_testing import (
    _test__ik_stream, _test__plan_free_motion_stream, _test__plan_holding_motion_stream, _test__sample_grasp_stream, 
    _test__sample_pose_stream, _test__test_cfree_approach_pose, _test__test_cfree_pose_pose, _test__test_cfree_traj_pose
)

# Imports that have two different implementations
from examples.fetch.from_kuka_tamp.utils.utils import PybulletToolsVersion

def _version_specific_imports(version:PybulletToolsVersion):
    from examples.fetch.from_kuka_tamp.utils.utils import UTILS, import_from, import_module
    pybullet_tools = import_module("pybullet_tools", UTILS[version])
    motion = import_module("motion", UTILS[version])
    return (
        pybullet_tools, motion, 
        # from pybullet_tools.utils import Point, Pose, get_sample_fn, WorldSaver, 
        *import_from("utils", targets=["Point", "Pose", "get_sample_fn", "WorldSaver"], package=pybullet_tools),
        # from pybullet_tools.kuka_primitives import Attach, BodyPath, Command, BodyConf, BodyPose
        *import_from("kuka_primitives", targets=["Attach", "BodyPath", "Command", "BodyConf", "BodyPose"], package=pybullet_tools)              
    )

VERSION = PybulletToolsVersion.PDDLSTREAM
pybullet_tools, motion, Point, pose, get_sample_fn, WorldSaver, Attach, BodyPath, Command, BodyConf, BodyPose = _version_specific_imports(VERSION)

###################### MAIN FUNCTIONALITY #########################

def pddlstream_from_problem(sim:MyiGibsonSemanticInterface, movable=[], teleport=False, grasp_name='top'):
    domain_pddl = read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'domain.pddl'))
    stream_pddl = read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stream.pddl'))
    constant_map = {}

    robot = sim.robot
    objects = sim.objects 
    movable = [obj for obj in objects if sim.is_movable(obj)]
    fixed = [obj for obj in objects if not sim.is_movable(obj)]

    # bid-based 
    robot = sim.robot_bid
    movable = [sim.get_bid(obj) for obj in movable]
    fixed = [sim.get_bid(obj) for obj in fixed]

    conf = BodyConf(robot, sim.get_arm_config())
    init = [('CanMove',),
            ('Conf', conf),
            ('AtConf', conf),
            ('HandEmpty',)]
    
    for body in movable:
        pose = BodyPose(body, sim.get_pose(body)) # body.get_base_link_position_orientation()
        init += [('Graspable', body),
                 ('Pose', body, pose),
                 ('AtPose', body, pose)]
        
        for surface in fixed:
            if surface not in ['walls', 'ceilings']:
                init += [('Stackable', body, surface)]
                if sim.is_on(body, surface):
                    init += [('Supported', body, pose, surface)]

    init += [('Sink', sim.get_bid('sink'))]
    init += [('Stove', sim.get_bid('stove'))]
    
    body = movable[0]
    goal_conf = BodyConf(robot, sim.sample_arm_config(), sim._arm_joint_ids)
    goal = ('and',
            ('AtConf', goal_conf),
            # ('Cooked', body),
    )

    stream_map = {
        'sample-pose': from_gen_fn(sim.get_stable_gen()),
        'sample-grasp': from_gen_fn(sim.get_grasp_gen()),
        'inverse-kinematics': from_fn(sim.get_grasp_traj_fn()),
        'plan-free-motion': from_fn(sim.get_free_motion_gen()),
        'plan-holding-motion': from_fn(sim.get_motion_gen()),

        'test-cfree-pose-pose': from_test(sim.get_cfree_pose_pose_test()),
        'test-cfree-approach-pose': from_test(sim.get_cfree_approach_obj_pose_test()),
        'test-cfree-traj-pose': from_test(negate_test(sim.get_command_obj_pose_collision_test())),

        # 'TrajCollision': sim.get_command_obj_pose_collision_test()
    }
    # _test_streams(sim, stream_map, init, streams='all')

    # stream_map = kuka_stream_map(sim, fixed=['floor','sink','stove'])
    
    
    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)





def main():
    parser = create_parser()
    parser.add_argument('-g','--gui', action='store_true', help='Render and visualize the system')
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = "fetch_tamp.yaml"
    config_path = os.path.join(dir_path,config)

    objects = [
        {'category':'sink',     'position':Point(-0.5,0,0), 'fixed_base':True}, 
        {'category':'stove',    'position':Point(+0.5,0,0), 'fixed_base':True}, 
        {'category':'celery',   'position':Point(0,+0.5,0), 'fixed_base':False}, 
        {'category':'radish',   'position':Point(0,-0.5,0), 'fixed_base':False}
    ]
    
    movable = ['celery', 'radish']
    
    sim = MyiGibsonSemanticInterface(config_file=config_path, 
                                         objects=objects,
                                         headless=(not args.gui)
                                        )
    sim.env.land(sim.get_object('celery'), Point(0,+0.5,0), (0,0,0))
    
    print(f"Objects: {sim.objects}")
    saver = WorldSaver()

    problem = pddlstream_from_problem(sim=sim, movable=movable)
    _, _, _, stream_map, init, goal = problem

    print('Init:')
    print(fmt_fluents(init))
    print('Goal:', goal)
    print('Streams:', str_from_object(set(stream_map)))
    
    print()
    fluents = get_fluents(init, 'AtConf')
    print('INITIAL CONFIG: ')
    print_fluents(fluents)
    print('GOAL CONFIG: ')
    print_fluents(goal)
    print()




    try:
        # with Profiler():
        #     with LockRenderer(sim, lock=True): # disable rendering during object loading
        #         solution = solve(
        #             problem, 
        #             algorithm=args.algorithm, 
        #             unit_costs=True, 
        #             success_cost=INF,
        #             debug=True
        #         )
        #         saver.restore()
        solution = solve(
            problem, 
            algorithm=args.algorithm, 
            unit_costs=True, 
            success_cost=INF,
            debug=True
        )
        saver.restore()
        print(solution)
        # print_solution(solution)
        # plan, cost, evaluations = solution
    finally:
        sim.close()





#########################################################




def _test_streams(sim:MyiGibsonSemanticInterface, stream_map:Dict[str,Callable], state:List, streams:Union[str,Iterable[str]]='all'):
    if streams == 'all':
        streams = stream_map.keys()
    elif isinstance(streams,str):
        streams = [str]
    target = "celery"

    if 'sample-pose' in streams:
        _test__sample_pose_stream(stream_map, target, surface="stove")
    if 'sample-grasp' in streams:
        _test__sample_grasp_stream(sim, stream_map, target)
    if 'inverse-kinematics' in streams:
        _test__ik_stream(sim, stream_map, target)
    if 'plan-free-motion' in streams:
        _test__plan_free_motion_stream(sim, stream_map, state, target)
    if 'plan-holding-motion' in streams:
        _test__plan_holding_motion_stream(sim, stream_map, state, target)
    if 'test-cfree-pose-pose' in streams:
        _test__test_cfree_pose_pose(stream_map, target, b2="radish")
    if 'test-cfree-approach-pose' in streams:
        _test__test_cfree_approach_pose(sim, stream_map, target)
    if 'test-cfree-traj-pose' in streams: # this one basically handles 'TrajCollision' too
        _test__test_cfree_traj_pose(sim, stream_map, obstacle=target)


def kuka_stream_map(sim:MyiGibsonSemanticInterface, fixed:List):
    from utils.utils import import_from
    get_stable_gen, get_grasp_gen, get_ik_fn, get_free_motion_gen, get_holding_motion_gen, get_movable_collision_test = \
        import_from("kuka_primitives", targets=[
            "get_stable_gen", 
            "get_grasp_gen", 
            "get_ik_fn", 
            "get_free_motion_gen", 
            "get_holding_motion_gen", 
            "get_movable_collision_test"
        ], package=pybullet_tools)      
    get_cfree_pose_pose_test, get_cfree_obj_approach_pose_test = import_from(
        "streams", 
        targets=["get_cfree_pose_pose_test", "get_cfree_obj_approach_pose_test"], 
        package="examples.pybullet.tamp"
    )
    
    robot = sim.robot_bid
    fixed = [1, 2, 3]
    teleport=False
    grasp_name='top'
    stream_map = {
        'sample-pose': from_gen_fn(get_stable_gen(fixed)),
        'sample-grasp': from_gen_fn(get_grasp_gen(robot, grasp_name)),
        'inverse-kinematics': from_fn(get_ik_fn(robot, fixed, teleport)),
        'plan-free-motion': from_fn(get_free_motion_gen(robot, fixed, teleport)),
        'plan-holding-motion': from_fn(get_holding_motion_gen(robot, fixed, teleport)),

        'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test()),
        'test-cfree-approach-pose': from_test(get_cfree_obj_approach_pose_test()),
        'test-cfree-traj-pose': from_test(negate_test(get_movable_collision_test())), #get_cfree_traj_pose_test()),

        'TrajCollision': get_movable_collision_test(),
    }
    return stream_map


def print_fluents(fluents):
    for i, fluent in enumerate(fluents):
        if fluent[0].lower() in ['pose', 'atpose']:
            fluent=(*fluent[:-1], tuple(tuple(np.round(x,3)) for x in fluent[-1].pose))
        elif fluent[0].lower() in ['conf', 'atconf']:
            fluent=(*fluent[:-1], tuple(np.round(fluent[-1].configuration,3)))
        print(i, fluent)

def get_fluents(fluents, name):
    return [fluent for fluent in fluents if fluent[0].lower()==name.lower()]

if __name__ == '__main__':
    main()