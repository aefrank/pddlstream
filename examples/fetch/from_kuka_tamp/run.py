#!/usr/bin/env python

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

from examples.fetch.from_kuka_tamp.fetch_primitives import MyiGibsonSemanticInterface, is_placement
from examples.pybullet.utils.pybullet_tools.utils import Point, Pose, stable_z

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
    fixed, movable = partition(lambda obj: ig.is_movable(obj), objects)

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

    [print(state) for state in init]












    








#######################################################


def main():
    headless = True

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
                                         headless=headless
                                        )

        


    pddlstream_from_problem(ig=iGibson, movable=movable)
   
    iGibson.env.close()

    


if __name__ == '__main__':
    main()