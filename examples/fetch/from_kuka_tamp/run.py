#!/usr/bin/env python

import sys
from typing import Any, Iterable, List, Optional, Tuple, Union

import argparse
import os
import functools as ft
import numpy as np

from examples.fetch.from_kuka_tamp.utils \
    import is_nonstring_iterable, is_numeric, nonstring_iterable, recursive_map, recursive_map_advanced, round_numeric
from examples.fetch.from_kuka_tamp.fetch_primitives import MyiGibsonSemanticInterface
from examples.pybullet.utils.pybullet_tools.utils \
    import Point, Pose, get_sample_fn, LockRenderer, WorldSaver
from examples.pybullet.utils.pybullet_tools.kuka_primitives \
    import Attach, BodyPath, Command
from pddlstream.algorithms.meta import solve
from pddlstream.utils \
    import INF, Profiler, negate_test, read, str_from_object
from pddlstream.language.constants \
    import PDDLProblem, print_solution
from pddlstream.language.generator \
    import from_fn, from_gen_fn, from_test

#######################################################


def pddlstream_from_problem(sim:MyiGibsonSemanticInterface, movable=[], teleport=False, grasp_name='top'):
    domain_pddl = read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'domain.pddl'))
    stream_pddl = read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stream.pddl'))
    constant_map = {}

    robot = sim.robot
    objects = sim.objects 
    movable = [obj for obj in objects if sim.is_movable(obj)]
    fixed = [obj for obj in objects if not sim.is_movable(obj)]

    conf = sim.get_arm_config()
    init = [('CanMove',),
            ('Conf', conf),
            ('AtConf', conf),
            ('HandEmpty',)]
    
    for body in movable:
        pose = (sim.get_position(body), sim.get_orientation(body)) # body.get_base_link_position_orientation()
        init += [('Graspable', body),
                 ('Pose', body, pose),
                 ('AtPose', body, pose)]
        
        for surface in fixed:
            init += [('Stackable', body, surface)]
            if sim.is_on(body, surface):
                init += [('Supported', body, pose, surface)]

    body = movable[0]
    goal = ('and',
            ('AtConf', conf),
            ('Cooked', body),
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

        'TrajCollision': sim.get_command_obj_pose_collision_test()
    }

    target = "celery"

    # _test__sample_pose_stream(stream_map, target, surface="stove")
    # _test__test_cfree_pose_pose(stream_map, target, b2="radish")
    # _test__sample_grasp_stream(sim, stream_map, target)
    # _test__ik_stream(sim, stream_map, target)
    # _test__plan_free_motion_stream(sim, stream_map, init, target)
    # _test__test_cfree_approach_pose(sim, stream_map, target)
    # _test__test_cfree_traj_pose(sim, stream_map, obstacle=target)
    # _test__plan_holding_motion_stream(sim, stream_map, init, target)
    
    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)



#######################################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g','--gui', action='store_true', help='Simulates the system')
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
    print(f"Objects: {sim.objects}")
    saver = WorldSaver()

    problem = pddlstream_from_problem(sim=sim, movable=movable)
    _, _, _, stream_map, init, goal = problem
    print('Init:')
    print(fmt_fluents(init))
    sys.exit()
                

    print('Goal:', goal)
    print('Streams:', str_from_object(set(stream_map)))

    try:
        with Profiler():
            with LockRenderer(lock=True): # disable rendering during object loading
                solution = solve(
                    problem, 
                    algorithm=args.algorithm, 
                    unit_costs=args.unit, 
                    success_cost=INF
                )
                saver.restore()
        print_solution(solution)
        plan, cost, evaluations = solution
    finally:
        sim.close()


def fmt_fluents(fluents):
    fn = lambda elem: round(elem,3) if is_numeric(elem) else elem
    fluent_str = "{"
    for fluent in fluents:
        label, *args = fluent
        args = recursive_map_advanced(fn, args, preserve_iterable_types=False, nonterminal_post_recursion_fn=tuple)
        fluent_str += str((label, *args)) + " "
    fluent_str += "}"
    return fluent_str

##########################  TEST STREAMS  ############################# 

def _test__plan_holding_motion_stream(sim:MyiGibsonSemanticInterface, stream_map:dict, state:List, target:str):
    atpose_fluents = [fluent for fluent in state if fluent[0].lower()=='atpose' and fluent[1].lower() != target]
    print("AtPose Fluents: ", [(name, obj, tuple(sim._fmt_num_iter(p) for p in pose)) for name,obj,pose in atpose_fluents])
    grasp = next(stream_map['sample-grasp'](target))[0][0]

    # Trajectory = returning to home position after having grasped object
    # grasp_pose -> approach_pose -> home pose
    p1 = grasp.grasp_pose
    q1 = sim.arm_ik(*p1)
    p2 = (grasp.approach_pose[0], None)
    q2 = sim.arm_ik(*p2)
    # Home position
    joint_vector_indices = sim._motion_planner.robot_arm_indices
    q3 = tuple(np.array(sim._robot.untucked_default_joint_pos)[joint_vector_indices])
    p3 = sim.arm_fk(q3)

    # print(f"Start:  Config:  {sim._fmt_num_iter(q1)} \tPose:  {'  '.join(sim._fmt_num_iter(p) for p in p1 if p is not None)}")
    # print(f"Goal:   Config:  {sim._fmt_num_iter(q2)} \tPose:  {'  '.join(sim._fmt_num_iter(p) for p in p2 if p is not None)}")
    # # print("Goal config: ", sim._fmt_num_iter(q2))
    # stream = stream_map['plan-holding-motion'](q1, q2, grasp, atpose_fluents=atpose_fluents)
    # sample = next(stream)[0]
    # print(sample)
    # command = sample[0]
    # print_Command(sim, command)

    # print("\n")
    
    print(f"Start:  Config:  {sim._fmt_num_iter(q2)} \tPose:  {'  '.join(sim._fmt_num_iter(p) for p in p2 if p is not None)}")
    print(f"Goal:   Config:  {sim._fmt_num_iter(q3)} \tPose:  {'  '.join(sim._fmt_num_iter(p) for p in p3 if p is not None)}")
    stream = stream_map['plan-holding-motion'](q2, q3, grasp, atpose_fluents=atpose_fluents)
    sample = next(stream)[0]
    print(sample)
    command = sample[0]
    print_Command(sim, command)

def _test__test_cfree_traj_pose(sim:MyiGibsonSemanticInterface, stream_map:dict, obstacle:str,
                                start_pose:Optional[Pose]=None, end_pose:Optional[Pose]=None):
    def get_command():
        q1 = sim.arm_ik(start_pose) if start_pose is not None else sim.get_arm_config()
        q2 = sim.arm_ik(end_pose)   if end_pose   is not None else (qrand:=get_sample_fn(sim.robot_id, sim._arm_joint_ids))()
        
        command_stream = stream_map['plan-free-motion'](q1,q2)
        command = next(command_stream)[0]
        if len(command) < 1:
            if end_pose is not None:
                raise RuntimeError("f'plan-free-motion' stream could not find a path for start and end poses {start_pose} and{end_pose}")
            else:
                for _ in range(1000):
                    q2 = qrand()
                    command_stream = stream_map['plan-free-motion'](q1,q2)
                    command = next(command_stream)[0]
                    if len(command) >= 1:
                        break
        command = command[0]
        return command
    
    command = get_command()
    assert isinstance(command,Command), f"command '{command}' is not of type Command"
    print_Command(sim, command)

    obstacle_pose = sim.get_pose(obstacle)
    test_stream = stream_map['test-cfree-traj-pose'](command, obstacle, obstacle_pose)
    result = next(test_stream)
    print(result)
    print_bool(result)

def _test__test_cfree_approach_pose(sim:MyiGibsonSemanticInterface, stream_map:dict, target:str):
    b1 = target
    p1 = sim.get_pose(b1)

    grasp_stream = stream_map['sample-grasp'](b1)
    g1 = next(grasp_stream)[0][0]
    print_BodyGrasp(sim, g1)

    test = stream_map['test-cfree-approach-pose']
    for b2 in sim.objects:
        p2 = sim.get_pose(b2)
        stream = test(b1,p1,g1,b2,p2)
        result = next(stream)
        print_bool(result)

def _test__plan_free_motion_stream(sim:MyiGibsonSemanticInterface, stream_map:dict, state:List, target:str):
    atpose_fluents = [fluent for fluent in state if fluent[0].lower()=='atpose']
    print("AtPose Fluents: ", [(name, obj, tuple(sim._fmt_num_iter(p) for p in pose)) for name,obj,pose in atpose_fluents])
    grasp = next(stream_map['sample-grasp'](target))[0][0]

    q1 = sim.get_arm_config()
    p1 = sim.get_pose(sim.eef_id)
    p2 = (grasp.approach_pose[0], None)
    q2 = sim.arm_ik(*p2)
    p3 = grasp.grasp_pose
    q3 = sim.arm_ik(*p3)

    print(f"Start:  Config:  {sim._fmt_num_iter(q1)} \tPose:  {'  '.join(sim._fmt_num_iter(p) for p in p1 if p is not None)}")
    print(f"Goal:   Config:  {sim._fmt_num_iter(q2)} \tPose:  {'  '.join(sim._fmt_num_iter(p) for p in p2 if p is not None)}")
    # print("Goal config: ", sim._fmt_num_iter(q2))
    stream = stream_map['plan-free-motion'](q1, q2, atpose_fluents=atpose_fluents)
    command = next(stream)[0][0]
    print_Command(sim, command)

    print("\n")
    
    print(f"Start:  Config:  {sim._fmt_num_iter(q2)} \tPose:  {'  '.join(sim._fmt_num_iter(p) for p in p2 if p is not None)}")
    print(f"Goal:   Config:  {sim._fmt_num_iter(q3)} \tPose:  {'  '.join(sim._fmt_num_iter(p) for p in p3 if p is not None)}")
    stream = stream_map['plan-free-motion'](q2, q3, atpose_fluents=atpose_fluents)
    command = next(stream)[0][0]
    print_Command(sim, command)

def _test__sample_pose_stream(stream_map:dict, target:str, surface:str):
    stream = stream_map['sample-pose'](target, surface)
    for _ in range(10):
        sample = next(stream)[0]
        pos, orn = sample.pose
        print(f"\nPosition:   \t{tuple(pos)}\nOrientation:   \t{tuple(orn)}")

def _test__test_cfree_pose_pose(stream_map:dict, b1:str, b2:str):
    stream = stream_map['test-cfree-pose-pose'](b1, body2=b2)
    print_bool(next(stream)[0])
    
def _test__sample_grasp_stream(sim:MyiGibsonSemanticInterface, stream_map:dict, target:str):
    stream = stream_map['sample-grasp'](target)
    for _ in range(10):
        grasp = next(stream)[0][0]
        print_BodyGrasp(sim, grasp)

def _test__ik_stream(sim, stream_map, target):
    grasp = next(stream_map['sample-grasp'](target))[0][0]
    print_BodyGrasp(sim, grasp)
    stream = stream_map['inverse-kinematics'](target, grasp)
    for _ in range(1):
        sample = next(stream)[0]
        if not sample:
            print(f"IK failed: {sample}")
            return
        config, command = sample
        print_Conf(sim, config)
        print_Command(sim, command)

        # print(f"\nConfiguration:   \t{[round(q,3) for q in config]}")
        # print(
        #     f"\nCommand <{command}>:"
        # )
        # for i,cmd in enumerate(command.body_paths):
        #     prefix = f"  {i}) {cmd.__class__.__name__}"
        #     if isinstance(cmd,BodyPath):
        #         string = f"\n\t - Body: {sim.get_name(cmd.body)}" \
        #                  f"\n\t - {len(cmd.path)} waypoints" \
        #                  f"\n\t - {len(cmd.joints)} joints" \
        #                  f"\n\t - Attachments: {[sim.get_name(g.body) for g in cmd.attachments]}"
        #     elif isinstance(cmd, Attach):
        #         string = f"\n\t - Robot: {sim.get_name(cmd.robot)}" \
        #                  f" (link={cmd.link})" \
        #                  f"\n\t - Target: {sim.get_name(cmd.body)}" \

                
        #     print(prefix+string)




def _temp_test(sim, target):
    q_start = sim.get_arm_config()
    print("Initial arm config: ", sim._fmt_num_iter(q_start))
    print("Direct-to-target config: ", sim._fmt_num_iter(sim.arm_ik(*sim.get_pose(target))))

    grasp_gen = sim.get_grasp_gen()(target)

    path = [None,None]
    while not all(path):
        q_approach = q_grasp = None
        while not all((q_approach, q_grasp)):
            grasp = next(grasp_gen)[0]
            print_BodyGrasp(sim, grasp)
            q_approach = sim.arm_ik(*grasp.approach_pose)
            q_grasp = sim.arm_ik(*grasp.grasp_pose)
        print("Approach config: ", sim._fmt_num_iter(q_approach))
        print("Grasp config:", sim._fmt_num_iter(q_grasp))

        sim.set_arm_config(q_start)
        path[0] = sim._motion_planner.plan_arm_motion(q_approach)
        sim.set_arm_config(q_approach)
        path[1] = sim._motion_planner.plan_arm_motion(q_grasp)
    try:
        print("Start-to-Approach:")
        print([sim._fmt_num_iter(q) for q in path[0]])   
    except:
        print("Start-to-Approach:")
        print(path[0])
    try:
        print("Approach-to-Grasp:")
        print([sim._fmt_num_iter(q) for q in path[1]])   
    except:
        print("Approach-to-Grasp:")
        print(path[1])

    # return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)


def print_bool(result:Union[bool,Iterable[bool]]) -> None:
    if result==[()]:
        print(f"Result: \tTrue")
    elif result==[]:
        print(f"Result: \tFalse")
    else:
        raise ValueError(f"Unexpected 'stream boolean' {result} of type {type(result)}. Expected [()]=True or []=False.")


def print_Conf(sim, conf):
    print(f"\nConfiguration:   \t{sim._fmt_num_iter(conf)}")

def print_BodyGrasp(sim, grasp):
    x,y,z = grasp.grasp_pose[0]
    a,b,c,d = grasp.grasp_pose[1]
    u,v,w = grasp.approach_pose[0]
    p,q,r,s = grasp.approach_pose[1]
    grasp_body = sim.get_name(grasp.body)
    grasp_robot = sim.get_name(grasp.robot)
    print(
        f"\nBody Grasp <{grasp}>:"
        f"\n  - Body:           name: {grasp_body}\tBID: {grasp.body}"
        f"\n  - Grasp Pose:     Position: ({x:.2f},{y:.2f},{z:.2f})    \tOrientation: ({a:.2f},{b:.2f},{c:.2f},{d:.2f}) "
        f"\n  - Approach pose:  Position: ({u:.2f},{v:.2f},{w:.2f})    \tOrientation: ({p:.2f},{q:.2f},{r:.2f},{s:.2f}) "
        f"\n  - Robot:          name: {grasp_robot}\tBID: {grasp.robot}"
        f"\n  - Link:           {grasp.link}"
        f"\n  - Index:          {grasp.index}"
    )

def print_Command(sim, command):
    print(
        f"\nCommand <{command}>:"
    )
    for i,cmd in enumerate(command.body_paths):
        prefix = f"  {i}) {cmd.__class__.__name__}"
        if isinstance(cmd,BodyPath):
            string = f"\n\t - Body: {sim.get_name(cmd.body)}" \
                        f"\n\t - {len(cmd.path)} waypoints" \
                        f"\n\t - {len(cmd.joints)} joints" \
                        f"\n\t - Attachments: {[sim.get_name(g.body) for g in cmd.attachments]}"
        elif isinstance(cmd, Attach):
            string = f"\n\t - Robot: {sim.get_name(cmd.robot)}" \
                        f" (link={cmd.link})" \
                        f"\n\t - Target: {sim.get_name(cmd.body)}" \

            
        print(prefix+string)

if __name__ == '__main__':
    main()