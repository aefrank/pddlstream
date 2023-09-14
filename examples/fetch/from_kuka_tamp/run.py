#!/usr/bin/env python

import argparse
from itertools import filterfalse, tee
import os
import sys
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
# import igibson
# from igibson import object_states
# from igibson.envs.igibson_env import iGibsonEnv
# from igibson.objects.articulated_object import URDFObject
# from igibson.objects.object_base import BaseObject
# from igibson.objects.ycb_object import YCBObject
# from igibson.robots import BaseRobot, Fetch
# from igibson.simulator import Simulator
# from igibson.utils.assets_utils import get_all_object_categories, get_ig_avg_category_specs, get_ig_model_path, get_object_models_of_category
# from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
# from igibson.utils.utils import parse_config

from examples.fetch.from_kuka_tamp.fetch_primitives import MyiGibsonSemanticInterface
from examples.pybullet.utils.pybullet_tools.utils import Point, Pose, get_sample_fn, stable_z
from examples.pybullet.utils.pybullet_tools.kuka_primitives import Attach, BodyPath, Command, get_ik_fn
from pddlstream.language.generator import from_fn, from_gen_fn, from_test

from pddlstream.utils import negate_test, read

#######################################################


def partition(pred, iterable):
    '''Use a predicate to partition entries into false entries and true entries
    
    partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    (from https://docs.python.org/3/library/itertools.html#itertools-recipes)
    '''
    t1, t2 = tee(iterable)
    return filterfalse(pred, t1), filter(pred, t2)    

# def get_identifier_and_pyobj(env:iGibsonEnv, obj:Union[BaseObject,str,int], identifier="name") -> Tuple[Union[str,int], BaseObject]:
#     if isinstance(obj, str):   obj = env.scene.objects_by_name[obj]
#     elif isinstance(obj, int): obj = env.scene.objects_by_id[obj]
#     if identifier=="name":
#         return obj.name, obj
#     elif identifier=="id" or identifier=="body_id":
#         return obj.get_body_ids(), obj
#     else: 
#         raise ValueError(f"argument 'id' expected values 'name', 'id', or 'body_id'. Instead received {id}")

def pddlstream_from_problem(ig:MyiGibsonSemanticInterface, movable=[], teleport=False, grasp_name='top'):
    domain_pddl = read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'domain.pddl'))
    stream_pddl = read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stream.pddl'))
    constant_map = {}

    robot = ig.robots[0]
    objects = ig.objects 
    movable = [obj for obj in objects if ig.is_movable(obj)]
    fixed = [obj for obj in objects if not ig.is_movable(obj)]

    conf = ig.get_joint_states(robot)
    init = [('CanMove',),
            ('Conf', conf),
            ('AtConf', conf),
            ('HandEmpty',)]
    
    for body in movable:
        pose = (ig.get_position(body), ig.get_orientation(body)) # body.get_base_link_position_orientation()
        init += [('Graspable', body),
                 ('Pose', body, pose),
                 ('AtPose', body, pose)]
        
        for surface in fixed:
            init += [('Stackable', body, surface)]
            if ig.is_placement(body, surface):
                init += [('Supported', body, pose, surface)]

    body = movable[0]
    goal = ('and',
            ('AtConf', conf),
            ('Cooked', body),
    )

    stream_map = {
        'sample-pose': from_gen_fn(ig.get_stable_gen()),
        'sample-grasp': from_gen_fn(ig.get_grasp_gen()),
        'inverse-kinematics': from_fn(ig.get_grasp_traj_fn()),
        'plan-free-motion': from_fn(ig.get_free_motion_gen()),
        'plan-holding-motion': from_fn(ig.get_motion_gen()),

        'test-cfree-pose-pose': from_test(ig.get_cfree_pose_pose_test()),
        'test-cfree-approach-pose': from_test(ig.get_cfree_approach_obj_pose_test()),
        'test-cfree-traj-pose': from_test(negate_test(ig.get_command_obj_pose_collision_test())),

        'TrajCollision': ig.get_command_obj_pose_collision_test()
    }

    target = "celery"

    # _test__sample_pose_stream(stream_map, target, surface="stove")
    # _test__test_cfree_pose_pose(stream_map, target, b2="radish")
    # _test__sample_grasp_stream(ig, stream_map, target)
    # _test__ik_stream(ig, stream_map, target)
    # _test__plan_free_motion_stream(ig, stream_map, init, target)
    # _test__test_cfree_approach_pose(ig, stream_map, target)
    # _test__test_cfree_traj_pose(ig, stream_map, obstacle=target)
    _test__plan_holding_motion_stream(ig, stream_map, init, target)


def _test__plan_holding_motion_stream(ig:MyiGibsonSemanticInterface, stream_map:dict, state:List, target:str):
    atpose_fluents = [fluent for fluent in state if fluent[0].lower()=='atpose' and fluent[1].lower() != target]
    print("AtPose Fluents: ", [(name, obj, tuple(ig._fmt_num_iter(p) for p in pose)) for name,obj,pose in atpose_fluents])
    grasp = next(stream_map['sample-grasp'](target))[0][0]

    # Trajectory = returning to home position after having grasped object
    # grasp_pose -> approach_pose -> home pose
    p1 = grasp.grasp_pose
    q1 = ig.arm_ik(*p1)
    p2 = (grasp.approach_pose[0], None)
    q2 = ig.arm_ik(*p2)
    # Home position
    joint_vector_indices = ig._motion_planner.robot_arm_indices
    q3 = tuple(np.array(ig._robot.untucked_default_joint_pos)[joint_vector_indices])
    p3 = ig.arm_fk(q3)

    # print(f"Start:  Config:  {ig._fmt_num_iter(q1)} \tPose:  {'  '.join(ig._fmt_num_iter(p) for p in p1 if p is not None)}")
    # print(f"Goal:   Config:  {ig._fmt_num_iter(q2)} \tPose:  {'  '.join(ig._fmt_num_iter(p) for p in p2 if p is not None)}")
    # # print("Goal config: ", ig._fmt_num_iter(q2))
    # stream = stream_map['plan-holding-motion'](q1, q2, grasp, atpose_fluents=atpose_fluents)
    # sample = next(stream)[0]
    # print(sample)
    # command = sample[0]
    # print_Command(ig, command)

    # print("\n")
    
    print(f"Start:  Config:  {ig._fmt_num_iter(q2)} \tPose:  {'  '.join(ig._fmt_num_iter(p) for p in p2 if p is not None)}")
    print(f"Goal:   Config:  {ig._fmt_num_iter(q3)} \tPose:  {'  '.join(ig._fmt_num_iter(p) for p in p3 if p is not None)}")
    stream = stream_map['plan-holding-motion'](q2, q3, grasp, atpose_fluents=atpose_fluents)
    sample = next(stream)[0]
    print(sample)
    command = sample[0]
    print_Command(ig, command)



def _test__test_cfree_traj_pose(ig:MyiGibsonSemanticInterface, stream_map:dict, obstacle:str,
                                start_pose:Optional[Pose]=None, end_pose:Optional[Pose]=None):
    def get_command():
        q1 = ig.arm_ik(start_pose) if start_pose is not None else ig.get_arm_config()
        q2 = ig.arm_ik(end_pose)   if end_pose   is not None else (qrand:=get_sample_fn(ig.robot_id, ig._arm_joint_ids))()
        
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
    print_Command(ig, command)

    obstacle_pose = ig.get_pose(obstacle)
    test_stream = stream_map['test-cfree-traj-pose'](command, obstacle, obstacle_pose)
    result = next(test_stream)
    print(result)
    print_bool(result)

def _test__test_cfree_approach_pose(ig:MyiGibsonSemanticInterface, stream_map:dict, target:str):
    b1 = target
    p1 = ig.get_pose(b1)

    grasp_stream = stream_map['sample-grasp'](b1)
    g1 = next(grasp_stream)[0][0]
    print_BodyGrasp(ig, g1)

    test = stream_map['test-cfree-approach-pose']
    for b2 in ig.objects:
        p2 = ig.get_pose(b2)
        stream = test(b1,p1,g1,b2,p2)
        result = next(stream)
        print_bool(result)

def _test__plan_free_motion_stream(ig:MyiGibsonSemanticInterface, stream_map:dict, state:List, target:str):
    atpose_fluents = [fluent for fluent in state if fluent[0].lower()=='atpose']
    print("AtPose Fluents: ", [(name, obj, tuple(ig._fmt_num_iter(p) for p in pose)) for name,obj,pose in atpose_fluents])
    grasp = next(stream_map['sample-grasp'](target))[0][0]

    q1 = ig.get_arm_config()
    p1 = ig.get_pose(ig.eef_id)
    p2 = (grasp.approach_pose[0], None)
    q2 = ig.arm_ik(*p2)
    p3 = grasp.grasp_pose
    q3 = ig.arm_ik(*p3)

    print(f"Start:  Config:  {ig._fmt_num_iter(q1)} \tPose:  {'  '.join(ig._fmt_num_iter(p) for p in p1 if p is not None)}")
    print(f"Goal:   Config:  {ig._fmt_num_iter(q2)} \tPose:  {'  '.join(ig._fmt_num_iter(p) for p in p2 if p is not None)}")
    # print("Goal config: ", ig._fmt_num_iter(q2))
    stream = stream_map['plan-free-motion'](q1, q2, atpose_fluents=atpose_fluents)
    command = next(stream)[0][0]
    print_Command(ig, command)

    print("\n")
    
    print(f"Start:  Config:  {ig._fmt_num_iter(q2)} \tPose:  {'  '.join(ig._fmt_num_iter(p) for p in p2 if p is not None)}")
    print(f"Goal:   Config:  {ig._fmt_num_iter(q3)} \tPose:  {'  '.join(ig._fmt_num_iter(p) for p in p3 if p is not None)}")
    stream = stream_map['plan-free-motion'](q2, q3, atpose_fluents=atpose_fluents)
    command = next(stream)[0][0]
    print_Command(ig, command)

def _test__sample_pose_stream(stream_map:dict, target:str, surface:str):
    stream = stream_map['sample-pose'](target, surface)
    for _ in range(10):
        sample = next(stream)[0]
        pos, orn = sample.pose
        print(f"\nPosition:   \t{tuple(pos)}\nOrientation:   \t{tuple(orn)}")

def _test__test_cfree_pose_pose(stream_map:dict, b1:str, b2:str):
    stream = stream_map['test-cfree-pose-pose'](b1, body2=b2)
    print_bool(next(stream)[0])
    
def _test__sample_grasp_stream(ig:MyiGibsonSemanticInterface, stream_map:dict, target:str):
    stream = stream_map['sample-grasp'](target)
    for _ in range(10):
        grasp = next(stream)[0][0]
        print_BodyGrasp(ig, grasp)

def _test__ik_stream(ig, stream_map, target):
    grasp = next(stream_map['sample-grasp'](target))[0][0]
    print_BodyGrasp(ig, grasp)
    stream = stream_map['inverse-kinematics'](target, grasp)
    for _ in range(1):
        sample = next(stream)[0]
        if not sample:
            print(f"IK failed: {sample}")
            return
        config, command = sample
        print_Conf(ig, config)
        print_Command(ig, command)

        # print(f"\nConfiguration:   \t{[round(q,3) for q in config]}")
        # print(
        #     f"\nCommand <{command}>:"
        # )
        # for i,cmd in enumerate(command.body_paths):
        #     prefix = f"  {i}) {cmd.__class__.__name__}"
        #     if isinstance(cmd,BodyPath):
        #         string = f"\n\t - Body: {ig.get_name(cmd.body)}" \
        #                  f"\n\t - {len(cmd.path)} waypoints" \
        #                  f"\n\t - {len(cmd.joints)} joints" \
        #                  f"\n\t - Attachments: {[ig.get_name(g.body) for g in cmd.attachments]}"
        #     elif isinstance(cmd, Attach):
        #         string = f"\n\t - Robot: {ig.get_name(cmd.robot)}" \
        #                  f" (link={cmd.link})" \
        #                  f"\n\t - Target: {ig.get_name(cmd.body)}" \

                
        #     print(prefix+string)



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
    
    iGibson = MyiGibsonSemanticInterface(config_file=config_path, 
                                         objects=objects,
                                         headless=(not args.gui)
                                        )

        


    pddlstream_from_problem(ig=iGibson, movable=movable)
   
    iGibson.env.close()




def _temp_test(ig, target):
    q_start = ig.get_arm_config()
    print("Initial arm config: ", ig._fmt_num_iter(q_start))
    print("Direct-to-target config: ", ig._fmt_num_iter(ig.arm_ik(*ig.get_pose(target))))

    grasp_gen = ig.get_grasp_gen()(target)

    path = [None,None]
    while not all(path):
        q_approach = q_grasp = None
        while not all((q_approach, q_grasp)):
            grasp = next(grasp_gen)[0]
            print_BodyGrasp(ig, grasp)
            q_approach = ig.arm_ik(*grasp.approach_pose)
            q_grasp = ig.arm_ik(*grasp.grasp_pose)
        print("Approach config: ", ig._fmt_num_iter(q_approach))
        print("Grasp config:", ig._fmt_num_iter(q_grasp))

        ig.set_arm_config(q_start)
        path[0] = ig._motion_planner.plan_arm_motion(q_approach)
        ig.set_arm_config(q_approach)
        path[1] = ig._motion_planner.plan_arm_motion(q_grasp)
    try:
        print("Start-to-Approach:")
        print([ig._fmt_num_iter(q) for q in path[0]])   
    except:
        print("Start-to-Approach:")
        print(path[0])
    try:
        print("Approach-to-Grasp:")
        print([ig._fmt_num_iter(q) for q in path[1]])   
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


def print_Conf(ig, conf):
    print(f"\nConfiguration:   \t{ig._fmt_num_iter(conf)}")

def print_BodyGrasp(ig, grasp):
    x,y,z = grasp.grasp_pose[0]
    a,b,c,d = grasp.grasp_pose[1]
    u,v,w = grasp.approach_pose[0]
    p,q,r,s = grasp.approach_pose[1]
    grasp_body = ig.get_name(grasp.body)
    grasp_robot = ig.get_name(grasp.robot)
    print(
        f"\nBody Grasp <{grasp}>:"
        f"\n  - Body:           name: {grasp_body}\tBID: {grasp.body}"
        f"\n  - Grasp Pose:     Position: ({x:.2f},{y:.2f},{z:.2f})    \tOrientation: ({a:.2f},{b:.2f},{c:.2f},{d:.2f}) "
        f"\n  - Approach pose:  Position: ({u:.2f},{v:.2f},{w:.2f})    \tOrientation: ({p:.2f},{q:.2f},{r:.2f},{s:.2f}) "
        f"\n  - Robot:          name: {grasp_robot}\tBID: {grasp.robot}"
        f"\n  - Link:           {grasp.link}"
        f"\n  - Index:          {grasp.index}"
    )

def print_Command(ig, command):
    print(
        f"\nCommand <{command}>:"
    )
    for i,cmd in enumerate(command.body_paths):
        prefix = f"  {i}) {cmd.__class__.__name__}"
        if isinstance(cmd,BodyPath):
            string = f"\n\t - Body: {ig.get_name(cmd.body)}" \
                        f"\n\t - {len(cmd.path)} waypoints" \
                        f"\n\t - {len(cmd.joints)} joints" \
                        f"\n\t - Attachments: {[ig.get_name(g.body) for g in cmd.attachments]}"
        elif isinstance(cmd, Attach):
            string = f"\n\t - Robot: {ig.get_name(cmd.robot)}" \
                        f" (link={cmd.link})" \
                        f"\n\t - Target: {ig.get_name(cmd.body)}" \

            
        print(prefix+string)

if __name__ == '__main__':
    main()