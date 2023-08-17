from collections import UserDict
from dataclasses import dataclass
import functools as ft
from itertools import filterfalse, tee
import os
import random
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from typing_extensions import TypeAlias

from igibson import object_states
from igibson.objects.object_base import BaseObject
from igibson.objects.stateful_object import StatefulObject
from igibson.envs.igibson_env import iGibsonEnv
from igibson.robots import BaseRobot
from igibson.scenes.igibson_indoor_scene import URDFObject
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_all_object_categories, get_ig_avg_category_specs, get_ig_model_path, get_object_models_of_category
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
from igibson.utils.utils import parse_config, quatToXYZW, quaternions
from numpy.typing import ArrayLike
from transforms3d.euler import euler2quat

from examples.pybullet.utils.pybullet_tools.utils import Pose

Args: TypeAlias = list
Kwargs: TypeAlias = Dict[str,Any]

Real : TypeAlias = Union[int,float]
UniqueID : TypeAlias  = Union[str,int]

Position3D = NamedTuple("Position3D", x=Real, y=Real, z=Real)
Position2D = NamedTuple("Position2D", x=Real, y=Real)
Position : TypeAlias = Union[Position2D, Position3D]

Euler = NamedTuple("Euler", r=Real, p=Real, y=Real)
Quaternion : TypeAlias = Tuple[Real, Real, Real, Real]
Orientation2D : TypeAlias = Real
Orientation3D : TypeAlias = Union[Euler, Quaternion]
Orientation : TypeAlias = Union[Orientation2D, Orientation3D]

# def handle_object_name(*objargs):
#     def _get_obj_from_name(env,name)
    
#     def decorator(func):
#         @ft.wraps(func)
#         def wrapper(*args, **kwargs):
#             for oa in objargs:
#                 if isinstance(oa,int):
#                     args[oa]
#                 else:
#                     kwargs[oa]
#             return func(*args, **kwargs)

 

class iGibsonSemanticInterface:

    def __init__(self, env:iGibsonEnv):
        self._env = env

    @property
    def env(self) -> List[iGibsonEnv]:
        return self._env
    @property
    def _robots(self) -> List[BaseRobot]:
        return self.env.robots
    @property
    def robots(self) -> List[BaseRobot]:
        return [robot.name for robot in self.env.robots]
    @property
    def _objects(self) -> List[BaseObject]:
        return self.env.scene.get_objects()
    @property
    def objects(self) -> List[str]:
        return [obj.name for obj in self.env.scene.get_objects() if obj not in self.env.robots]

    # Object querying
    def _get_object(self, obj:Union[UniqueID,BaseObject]) -> BaseObject:
        assert isinstance(obj,(int,str,BaseObject)), f"Invalid argument 'obj' of type {type(obj)}: {obj}"
        if isinstance(obj, BaseObject): return obj
        elif isinstance(obj,int):       return self._get_object_from_bodyid(obj)
        elif isinstance(obj,str):       return self._get_object_from_name(obj)
    
    def _get_object_from_bodyid(self, body_id:int) -> BaseObject:
        assert isinstance(body_id,int), f"Invalid argument 'body_id' of type {type(body_id)}: {body_id}"
        return self.env.scene.objects_by_id[body_id]
    
    def _get_object_from_name(self, name:str) -> BaseObject:
        assert isinstance(name,str),  f"Invalid argument 'name' of type {type(name)}: {name}"
        return self.env.scene.objects_by_name[name]
    
    def _get_bodyids_from_name(self, name:str) -> List[int]:
        assert isinstance(name,str), f"Invalid argument 'name' of type {type(name)}: {name}"
        return self.env.scene.objects_by_name[name].get_body_ids()
    
    def _get_bodyids_from_obj(self, obj:BaseObject) -> List[int]:
        assert isinstance(obj,BaseObject), f"Invalid argument 'obj' of type {type(obj)}: {obj}"
        return obj.get_body_ids()
    
    def _get_name_from_bodyid(self, body_id:int) -> str:
        assert isinstance(body_id,int), f"Invalid argument 'body_id' of type {type(body_id)}: {body_id}"
        return self.env.scene.objects_by_id[body_id]


    # Position getter/setter
    def get_position(self, body:UniqueID) -> Position:
        return self._get_object(body).get_position()
    def set_position(self, body:UniqueID, position:Position) -> None:
        self._get_object(body).set_position(pos=position)

    # Orientation getter/setter
    def get_orientation(self, body:UniqueID) -> Orientation:
        return self._get_object(body).get_orientation()    
    def set_orientation(self, body:UniqueID, orientation:Orientation, *, force_quaternion=False) -> None:
        if force_quaternion and len(orientation)==3:
            orientation = quatToXYZW(euler2quat(*orientation), "wxyz")
        self._get_object(body).set_orientation(orn=orientation)

    # Combined Position and Orientation getter/setter
    def get_position_orientation(self, body:UniqueID) -> Tuple[Position, Orientation]:
        return self._get_object(body).get_position_orientation()
    def set_position_orientation(self, body:Union[str,int], 
                                 position:Position, 
                                 orientation:Orientation, 
                                 *, 
                                 force_quaternion=False) -> None:
        self.set_position(body=body, position=position)
        self.set_orientation(body=body, orientation=orientation, force_quaternion=force_quaternion)

    # Joint states
    def get_joint_states(self, body:UniqueID):
        return self._get_object(body).get_joint_states()

class URDFObjectSpec(UserDict):
    def __init__(self, category:str, name:Optional[str]=None, *, model:Optional[str]=None, **URDF_kwargs:Kwargs):
        if category not in get_all_object_categories():
            raise ValueError(f"Unable to find object category '{category}' in assets.")
        
        self.data = URDF_kwargs
        self.data["category"] = category
        self.data["name"] = name if (name is not None) else category

        if model is None: 
            model = random.choice(get_object_models_of_category(self.data["category"]))
        model_path = self.data["model_path"] if "model_path" in self.data else get_ig_model_path(self.data["category"], model)
        self.data["filename"] = os.path.join(model_path, model + ".urdf")
        
class ObjectSpec(UserDict):
    _states = ("position", "orientation")
    
    def __init__(self, position:Position=Position3D(0,0,0), orientation:Orientation=Euler(0,0,0), **URDF_kwargs):
        self.data = URDFObjectSpec(**URDF_kwargs)
        self.data.update({
            "position" : position,
            "orientation" : orientation
        })
    @property
    def urdf_data(self):
        return {k:v for k,v in self.data.items() if k not in ObjectSpec._states}
    @property
    def state_data(self):
        return {k:v for k,v in self.data.items() if k in ObjectSpec._states}
    @property
    def URDF(self):
        return URDFObjectSpec(**self.urdf_data)
    

class MyiGibsonSemanticInterface(iGibsonSemanticInterface):
    _states = ("position", "orientation")
    _state_access_map = {
        "position" : (
            lambda self, body: self.get_position(body), 
            lambda self, body, val: self.set_position(body,val)
        ),
        "orientation" : (
            lambda self, body: self.get_orientation(body), 
            lambda self, body,val: ft.partial(self.set_orientation, force_quaternion=True)(body,val)
        ),
    }
    assert set(_states) == set(_state_access_map)
    


    def __init__(self, config_file:str, objects:List[ObjectSpec], *, headless:bool=True, verbose=True):
        self._igibson_setup(config_file=config_file, headless=headless)
        self.load_objects(objects, verbose=verbose)
        self._init_object_state(objects,verbose=verbose)
        self._init_robot_state()

    
    def _init_object_state(self, obj_specs:List[ObjectSpec], *, verbose=True):
        for spec in obj_specs:
            assert isinstance(spec, dict)
            if not isinstance(spec,ObjectSpec): 
                spec = ObjectSpec(**spec)
            for state in self._states:
                if state in spec:    
                    self.set_state(spec["name"], state, spec[state])
            


    def is_movable(self, body: UniqueID):
        obj = self._get_object(body)
        return not obj.fixed_base

    # def get_joint_states(self, body: UniqueID):
        # only using joint position, not velocity
        # return [joint[0] for joint in super().get_joint_states(body)]


    def get_state(self, body:UniqueID, state:str):
        assert state in self._states
        return MyiGibsonSemanticInterface._state_access_map[state][0](self,body)
    def set_state(self, body:UniqueID, state:str, value:Any):
        assert state in self._states
        MyiGibsonSemanticInterface._state_access_map[state][1](self, body,value)


    def _igibson_setup(self, config_file:str, headless:bool=True) -> Tuple[iGibsonEnv, MotionPlanningWrapper]:
        env = iGibsonEnv(
            config_file=self._load_config(config_file=config_file),
            mode="gui_interactive" if not headless else "headless",
            action_timestep=1.0 / 120.0,
            physics_timestep=1.0 / 120.0,
        ) 
        self._env = env
        
        motion_planner = MotionPlanningWrapper(
            self.env,
            optimize_iter=10,
            full_observability_2d_planning=False,
            collision_with_pb_2d_planning=False,
            visualize_2d_planning=False,
            visualize_2d_result=False,
        )
        self._motion_planner = motion_planner

        if not headless:
            self._viewer_setup()


    def _load_config(self, config_file:str) -> dict:
        config_file = os.path.abspath(config_file)
        config_data = parse_config(config_file)

        config_data["load_object_categories"] = []  # accelerate loading (only building structures)
        config_data["visible_target"] = False
        config_data["visible_path"] = False

        # Reduce texture scale for Mac.
        if sys.platform == "darwin":
            config_data["texture_scale"] = 0.5
        
        self._config = config_data
        return self._config
    

    def _viewer_setup(self) -> None:
        # Set viewer camera 
        self.env.simulator.viewer.initial_pos = [-0.8, 0.7, 1.7]
        self.env.simulator.viewer.initial_view_direction = [0.1, -0.9, -0.5]
        self.env.simulator.viewer.reset_viewer()
        # Note: this was taken from iGibson motion planning example; 
        #       might want to refer to default camera used in pddlstream kuka 
        #       example instead if viewer pose is weird


    
    @classmethod
    def _as_urdf_spec(cls, spec:Union[str,dict]) -> URDFObjectSpec:
        # def _obj_attr_check(spec:dict) -> None:
        #     '''Ensure all necessary keys present in object dict, and no forbidden keys are present.
        #     '''
        #     necessary, forbidden = set(['category']), set(cls._states)
        #     missing = necessary - set(spec)
        #     extras = forbidden & set(spec)
        #     if len(missing)>0:  raise ValueError(f"object specification missing necessary key(s) '{missing}':\n{spec}")
        #     if len(extras)>0:   raise ValueError(f"object specification contains forbidden key(s) '{extras}':\n{spec}")        

        if isinstance(spec, dict) or isinstance(spec,UserDict):
            urdf_spec = ObjectSpec(**spec).URDF
        elif isinstance(spec,str): 
            urdf_spec = URDFObjectSpec(spec)
        else:
            raise TypeError(f"Inappropriate argument 'spec' of type {type(spec)}. Expected str or dict.")
        assert isinstance(urdf_spec, URDFObjectSpec)
        return urdf_spec



    def load_object(self, spec:Union[str,dict], verbose:bool=True) -> List[int] :
        URDF_kwargs = {
            'avg_obj_dims' : get_ig_avg_category_specs().get(spec["category"]),
            'fit_avg_dim_volume' : True,
            'texture_randomization' : False,
            'overwrite_inertial' : True,
        }
        
        spec = self._as_urdf_spec(spec)
        URDF_kwargs.update(**spec)

        if verbose: print(f"Loading {spec['category'].capitalize()} object '{spec['name']}'...", end='')
        
        # Create and import the object
        sim_obj = URDFObject(**URDF_kwargs)
        self.env.simulator.import_object(sim_obj)
        
        if verbose: print(" done.")
                
        # Return obj body_ids
        return sim_obj.get_body_ids()

    def load_objects(self, specifications:List[Union[str,dict]], verbose:bool=True) -> None:
        '''Load objects into simulator based on a list of categories.
        ''' 
        body_ids_by_object = {}
        for spec in specifications:
            spec = self._as_urdf_spec(spec)
            body_ids = self.load_object(spec=spec, verbose=verbose)
            body_ids_by_object[spec["name"]] = body_ids 
        return body_ids_by_object

            


    def _init_robot_state(self):
        '''Initialize robot state. 
        
        WARNING: this steps sim; no objects can be added to scene after this is called.
        '''
        self.env.reset()
        self.env.land(self.env.robots[0], [0, 0, 0], [0, 0, 0]) # note: this steps sim!
        for robot in self.env.robots:
            robot.tuck() # tuck arm 
        
    def is_placement(self, body:UniqueID, surface:UniqueID):
        body    = self._get_object(body)
        surface = self._get_object(body)
        return body.states[object_states.OnTop].get_value(surface)

    

def is_placement(body:StatefulObject, surface:StatefulObject):
    return body.states[object_states.OnTop].get_value(surface)

def get_stable_gen(fixed=[]):
    pass

def get_grasp_gen(robot, grasp_name='top'):
    pass

def get_tool_link(robot):
    pass

def get_ik_fn(robot, fixed=[], teleport=False, num_attempts=10):
    pass

def get_free_motion_gen(robot, fixed=[], teleport=False, self_collisions=True):
    pass

def get_holding_motion_gen(robot, fixed=[], teleport=False, self_collisions=True):
    pass

def get_movable_collision_test():
    pass

