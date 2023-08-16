#!/usr/bin/env python

import os
import random
import sys
from typing import List, Tuple, Union
import igibson
from igibson import object_states
from igibson.envs.igibson_env import iGibsonEnv
from igibson.objects.articulated_object import URDFObject
from igibson.robots import BaseRobot
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_all_object_categories, get_ig_avg_category_specs, get_ig_model_path, get_object_models_of_category
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
from igibson.utils.utils import parse_config
from numpy.typing import ArrayLike

import yaml
from examples.pybullet.utils.pybullet_tools.utils import Point, Pose, stable_z

from pddlstream.utils import read

#######################################################

def pddlstream_from_problem():
    domain_pddl = read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'domain.pddl'))
    stream_pddl = read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stream.pddl'))

def igibson_setup(config_file:str, headless:bool=True) -> Tuple[iGibsonEnv, MotionPlanningWrapper]:
    env = iGibsonEnv(
        config_file=config_setup(config_file=config_file),
        mode="headless",
        action_timestep=1.0 / 120.0,
        physics_timestep=1.0 / 120.0,
    ) 
    motion_planner = MotionPlanningWrapper(
        env,
        optimize_iter=10,
        full_observability_2d_planning=False,
        collision_with_pb_2d_planning=False,
        visualize_2d_planning=False,
        visualize_2d_result=False,
    )

    if not headless:
        viewer_setup(env=env)

    return env, motion_planner

def robot_setup(env:iGibsonEnv):
    env.reset()
    env.land(env.robots[0], [0, 0, 0], [0, 0, 0])
    for robot in env.robots:
        robot.tuck() # tuck arm

def viewer_setup(simulator: Simulator) -> None:
    # Set viewer camera 
    simulator.viewer.initial_pos = [-0.8, 0.7, 1.7]
    simulator.viewer.initial_view_direction = [0.1, -0.9, -0.5]
    simulator.viewer.reset_viewer()

def config_setup(config_file:str) -> dict:
    config_file = os.path.abspath(config_file)
    config_data = parse_config(config_file)

    config_data["load_object_categories"] = []  # accelerate loading (only building structures)
    config_data["visible_target"] = False
    config_data["visible_path"] = False

    # Reduce texture scale for Mac.
    if sys.platform == "darwin":
        config_data["texture_scale"] = 0.5
    
    return config_data

def obj_attr_check(obj:dict):
    necessary = set(['category'])
    forbidden = set([])
    
    # ensure all necessary keys present in object dict, and no forbidden keys are
    missing = necessary - set(obj)
    if len(missing)>0:
        raise ValueError(f"object dictionary missing necessary key(s) '{missing}':\n{obj}")
    extras = forbidden & set(obj)
    if len(extras)>0:
        raise ValueError(f"object dictionary contains forbidden key(s) '{extras}':\n{obj}")


def _load_object(simulator:Simulator, 
                 category:str, 
                 *, 
                 model:str=None, 
                 position:ArrayLike=None, 
                 orientation:ArrayLike=None, 
                 pose:Pose=None, 
                 verbose:bool=True,
                 **kwargs
                ) -> None:
    obj_name = category
    if verbose: print(f"Loading {category.capitalize()} object '{obj_name}'...", end='')

    assert not (pose and (position or orientation)), "cannot specify object 'position' or 'orientation' if 'pose' is specified"
    assert not (('model_path' in kwargs) and (model is not None)), "cannot specify both 'model' and 'model_path'"


    if 'model_path' not in kwargs:
        if model is None: 
            model = random.choice(get_object_models_of_category(category))
        model_path = get_ig_model_path(category, model)
        kwargs.update({'model_path':model_path})

    urdf_file = os.path.join(model_path, model + ".urdf")

    # Create and import the object
    urdf_kwargs = {
        'avg_obj_dims' : get_ig_avg_category_specs().get(category),
        'fit_avg_dim_volume' : True,
        'texture_randomization' : False,
        'overwrite_inertial' : True,
    }
    urdf_kwargs.update(**kwargs)
    sim_obj = URDFObject(
        urdf_file,
        category=category,
        name=obj_name,
        **urdf_kwargs
    )
    simulator.import_object(sim_obj)
    if verbose: print(" done.")

    if verbose: print(f"Setting object state...", end='')
    # Set pose if specified
    if pose is not None:            sim_obj.set_position_orientation(*pose)
    else:
        if position is not None:    sim_obj.set_position(pos=position)
        if orientation is not None: sim_obj.set_orientation(orn=orientation)
    if verbose: print(" done.\n")
    

def load_objects(simulator:Simulator, objects:List[Union[str,dict]], verbose:bool=True):
    for obj in objects:
        if isinstance(obj,str): 
            category = obj
            _load_object(simulator=simulator, category=obj, verbose=verbose)
        elif isinstance(obj, dict):
            obj_attr_check(obj)
            _load_object(simulator=simulator, verbose=verbose, **obj)
        else:
            raise TypeError(f"Inappropriate argument 'obj' of type {type(obj)}. Expected str or dict.")

       

def get_scene_objects(env:iGibsonEnv):
    robots = env.robots
    objects = env.scene.objects_by_name

    print()
    [print(f"Robot:\t{robot}") for robot in robots]
    print()
    [print(f"Object: {name}") for name,obj in objects.items() if not isinstance(obj, BaseRobot)]
    print()

#######################################################

def main():
    headless = True

    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = "fetch_tamp.yaml"
    env, motion_planner = igibson_setup(config_file=os.path.join(dir_path,config), headless=headless)

    objects = [
        {'category':'sink', 'pose':Pose(Point(-0.5,0,0)), 'fixed_base':True}, 
        {'category':'stove', 'pose':Pose(Point(+0.5,0,0)), 'fixed_base':True}, 
        {'category':'celery', 'pose':None, 'fixed_base':False}, 
        {'category':'radish', 'pose':None, 'fixed_base':False}
    ]

    
    try:        
        load_objects(simulator=env.simulator, objects=objects)
        # celerypos = Pose(Point(y=0.5, z=stable_z(celery, floor)))
        # radishpos = Pose(Point(y=-0.5, z=stable_z(radish, floor)))
        robot_setup(env=env)

        celery = env.scene.objects_by_name['celery']
        radish = env.scene.objects_by_name['radish']
        floor = env.scene.objects_by_name['floors']
        print(env.scene.floor_body_ids)
        get_scene_objects(env)
        f = floor.get_body_ids()[0]
        env.land(celery, pos=Point(y=0.5), orn=[0,0,0])
    finally:    
        env.close()


if __name__ == '__main__':
    main()