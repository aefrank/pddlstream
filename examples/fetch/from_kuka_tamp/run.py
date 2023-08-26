#!/usr/bin/env python

import argparse
from itertools import filterfalse, tee
import os
import sys
from typing import List, Tuple, Union
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
from examples.pybullet.utils.pybullet_tools.utils import Point, Pose, stable_z
from pddlstream.language.generator import from_fn, from_gen_fn, from_test

from pddlstream.utils import read

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

    # robot = Fetch().name]

    
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
        'sample-grasp': from_gen_fn(ig.get_grasp_gen(robot)),
        # 'inverse-kinematics': from_fn(get_ik_fn(robot, fixed, teleport)),
        # 'plan-free-motion': from_fn(get_free_motion_gen(robot, fixed, teleport)),
        # 'plan-holding-motion': from_fn(get_holding_motion_gen(robot, fixed, teleport)),

        'test-cfree-pose-pose': from_test(ig.get_cfree_pose_pose_test()),
        # 'test-cfree-approach-pose': from_test(get_cfree_obj_approach_pose_test()),
        # 'test-cfree-traj-pose': from_test(negate_test(get_movable_collision_test())), #get_cfree_traj_pose_test()),

        # 'TrajCollision': get_movable_collision_test(),
    }

    # _test__sample_pose_stream(stream_map)
    # _test__test_cfree_pose_pose(stream_map)
    _test__sample_grasp_stream(stream_map)
        

    # return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)



def _test__sample_pose_stream(stream_map):
    stream = stream_map['sample-pose']("celery", "stove")
    for _ in range(10):
        sample = next(stream)[0]
        pos, orn = sample
        print(f"\nPosition:   \t{tuple(pos)}\nOrientation:   \t{tuple(orn)}")

def _test__test_cfree_pose_pose(stream_map):
    stream = stream_map['test-cfree-pose-pose']("radish", body2="celery")
    print(next(stream)[0])
    

def _test__sample_grasp_stream(stream_map):
    stream = stream_map['sample-grasp']("celery")
    for _ in range(10):
        sample = next(stream)[0][0]
        x,y,z = sample.grasp_pose[0]
        a,b,c,d = sample.grasp_pose[1]
        u,v,w = sample.approach_pose[0]
        p,q,r,s = sample.approach_pose[1]
        print(
            f"\nBody Grasp <{sample}>:"
            f"\n  - Body:           {sample.body.name}"
            f"\n  - Grasp Pose:     Position: ({x:.2f},{y:.2f},{z:.2f})\t\tOrientation: ({a:.2f},{b:.2f},{c:.2f},{d:.2f}) "
            f"\n  - Approach pose:  Position: ({u:.2f},{v:.2f},{w:.2f})\t\tOrientation: ({p:.2f},{q:.2f},{r:.2f},{s:.2f}) "
            f"\n  - Robot:          {sample.robot.name} ({sample.robot.model_name})"
            f"\n  - Link:           {sample.link}"
            f"\n  - Index:          {sample.index}"
        )




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

    


if __name__ == '__main__':
    main()