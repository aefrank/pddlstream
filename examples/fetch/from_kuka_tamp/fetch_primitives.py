########################################################################################################################
##########################################    IMPORTS    ###############################################################
########################################################################################################################
from typing import \
    Any, Callable, Collection, Dict, NewType, Set, Iterable, List, Optional, Tuple, Type, Union, get_args
from typing_extensions import TypeAlias
from enum import Enum

import functools as ft
import os, sys, math
from itertools import product
import argparse

import cv2
import pybullet as pb
import numpy as np
from transforms3d.euler import euler2quat
from scipy.spatial.transform import Rotation



# iGibson modules
from igibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
from igibson.action_primitives.starter_semantic_action_primitives import \
    UndoableContext, GRASP_APPROACH_DISTANCE, MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE, PREDICATE_SAMPLING_Z_OFFSET
from igibson import object_states
from igibson.simulator import SimulatorMode

from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
from igibson.objects.object_base import BaseObject
from igibson.objects.articulated_object import URDFObject
from igibson.robots import BaseRobot
from igibson.robots.robot_base import RobotLink

from igibson.utils.utils import parse_config, quatToXYZW
from igibson.utils.assets_utils import get_ig_avg_category_specs
from igibson.object_states.utils import sample_kinematics
from igibson.utils.behavior_robot_motion_planning_utils import HAND_MAX_DISTANCE



# Custom modules
from iGibson.igibson.utils.grasp_planning_utils import get_hand_rotation_from_axes
from utils.object_spec import \
    Kwargs, ObjectSpec, Orientation, Position, Euler, _as_urdf_spec
from examples.fetch.from_kuka_tamp.utils.utils import \
    import_from, is_numeric_vector, nonstring_iterable, recursive_map_advanced, round_numeric, PybulletToolsVersion, import_module


# PYBULLET TOOLS 
# TODO: investigate differences between those found at examples.pybullet.utils.pybullet_tools vs igibson.external.pybullet_tools
from examples.pybullet.utils.pybullet_tools.kuka_primitives import \
    Attach, BodyConf, BodyGrasp, BodyPath, BodyPose, Command
from examples.pybullet.utils.pybullet_tools.utils import \
    Attachment, Saver, get_client, get_joint_positions, get_sample_fn, interpolate_poses, quat_from_euler
from igibson.external.pybullet_tools.utils import \
    all_between, get_aabb, get_custom_limits, get_moving_links, get_self_link_pairs, is_collision_free, link_from_name, \
    pairwise_collision, pairwise_link_collision, set_joint_positions


###################   Disambiguating different versions of utils   #########################
# Pybullet Planning implementation: original by Caelan Reed Garrett vs. iGibson version 
# TODO: Incorporate other possibly-ambiguous modules 
VERSION = PybulletToolsVersion.PDDLSTREAM
pybullet_tools = import_module("pybullet_tools", VERSION)
motion = import_module("motion", VERSION)

# PybulletToolsVersion = Enum('PybulletToolsVersion', ['PDDLSTREAM', 'IGIBSON'])
# STREAM = PybulletToolsVersion.PDDLSTREAM
# IGIBSON = PybulletToolsVersion.IGIBSON
# UTILS = {
#     STREAM : 'examples.pybullet.utils',
#     IGIBSON : 'igibson.external'
# }
# PYBULLET_TOOLS_MODULES = {
#     STREAM :  importlib.import_module('.pybullet_tools', UTILS[STREAM]),
#     IGIBSON : importlib.import_module('.pybullet_tools', UTILS[IGIBSON]),
# }
# MOTION_PLANNING_MODULES = {
#     STREAM:   importlib.import_module('.motion.motion_planners', UTILS[STREAM]),
#     IGIBSON : importlib.import_module('.motion.motion_planners', UTILS[IGIBSON]),
# }

################# Custom types for slightly more "self-documenting" code #######################
# T = TypeVar('T')

BID = NewType('BID', int) # body_id that can be used with pybullet/pybullet_tools
JID = NewType('JID', int) # link/joint id that can be used with pybullet/pybullet_tools

UID: TypeAlias = Union[str,int] # any kind of concise name/id that uniquely identifies something
Object: TypeAlias = Union[UID,BaseObject]
Robot: TypeAlias = Union[UID,BaseRobot]

JointPos = Iterable[float]
Pose: TypeAlias = Tuple[Position,Orientation]
# Arm kinematic specification/constraint
KinematicConstraint: TypeAlias = Union[BodyConf,BodyPose,JointPos,Pose,Position,Orientation]

class iGibsonSemanticInterface:

    def __init__(self, env:iGibsonEnv):
        self._env = env
    def close(self):
        self._env.close()

    ################################ Basic data/component access ################################
    @property
    def env(self) -> iGibsonEnv:
        return self._env
    @property
    def _robots(self) -> List[BaseRobot]:
        return self.env.robots
    @property
    def robots(self) -> List[str]:
        return [robot.name for robot in self.env.robots]
    @property
    def _objects(self) -> List[BaseObject]:
        return self.env.scene.get_objects()
    @property
    def objects(self) -> List[str]:
        return [obj.name for obj in self.env.scene.get_objects() if obj not in self.env.robots]
    
    ################################ Object query handling ################################
    # Translating between object names (str), pybullet body ids (BID), and python objects (BaseObject)
    def get_bid(self, obj:Object) -> BID:
        return self._get_bids(obj)[0]
    def _get_bids(self, obj:Object) -> List[BID]:
        if isinstance(obj,BaseObject): return obj.get_body_ids()
        elif isinstance(obj,str): return self._get_bodyids_from_name(obj)
        elif isinstance(obj,int):
            assert obj in self.env.scene.objects_by_id, f"invalid body_id {obj}; not found in simulator"
            return [obj]
        else:
            raise TypeError(
                f"Inappropriate argument obj={obj} of type {type(obj)}. " 
                "Expected BaseObject, str, or int"
            )
    def get_name(self, obj:Object) -> BaseObject:
        if isinstance(obj, BaseObject): return self._get_name_from_obj(obj)
        elif isinstance(obj,int):       return self._get_name_from_bodyid(obj)
        elif isinstance(obj,str):       
            assert obj in self.env.scene.objects_by_name, f"invalid object name '{obj}'"
            return obj
        else:
            raise TypeError(
                f"Inappropriate argument obj={obj} of type {type(obj)}. " 
                "Expected BaseObject, str, or int"
            )
    def get_object(self, obj:Object) -> BaseObject:
        if isinstance(obj, BaseObject): 
            assert obj in self._objects, f"object '{obj}' not found in current scene"
            return obj
        elif isinstance(obj,int):       return self._get_object_from_bodyid(obj)
        elif isinstance(obj,str):       return self._get_object_from_name(obj)
        else:
            raise TypeError(
                f"Inappropriate argument obj={obj} of type {type(obj)}. " 
                "Expected BaseObject, str, or int"
            )

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
        return self.env.scene.objects_by_id[body_id].name
    def _get_name_from_obj(self, obj:BaseObject) -> str:
        assert isinstance(obj,BaseObject), f"Invalid argument 'obj' of type {type(obj)}: {obj}"
        return obj.name
        
    ################################ Object State Handling ################################
    # Position getter/setter
    def get_position(self, body:UID) -> Position:
        return self.get_object(body).get_position()
    def set_position(self, body:UID, position:Position) -> None:
        self.get_object(body).set_position(pos=position)

    # Orientation getter/setter
    def get_orientation(self, body:UID) -> Orientation:
        return self.get_object(body).get_orientation()    
    def set_orientation(self, body:UID, orientation:Orientation, *, force_quaternion=True) -> None:
        if force_quaternion and len(orientation)==3:
            orientation = quatToXYZW(euler2quat(*orientation), "wxyz")
        self.get_object(body).set_orientation(orn=orientation)

    # Combined Position and Orientation getter/setter
    def get_position_orientation(self, body:UID) -> Pose:
        return self.get_object(body).get_position_orientation()
    def set_position_orientation(self, body:UID, 
                                 position:Position, 
                                 orientation:Orientation, 
                                 *, 
                                 force_quaternion=True) -> None:
        self.set_position(body=body, position=position)
        self.set_orientation(body=body, orientation=orientation, force_quaternion=force_quaternion)

    # Joint config getter/setters
    def get_joint_positions(self, body:UID, joints:Iterable[int]) -> JointPos:
        return get_joint_positions(body, joints)
    def set_joint_positions(self, body:UID, joints:Iterable[int], values:JointPos):
        return set_joint_positions(body, joints, values)
    def get_joint_states(self, body:UID) -> Dict[str,Iterable[Tuple[float,float]]]:
        '''Returned as dictionary {"joint_name":(q, qdot)}'''
        return self.get_object(body).get_joint_states()
    
################################################################################################
################################################################################################

# def _sync_viewer_after_exec(mthd):
#         @ft.wraps(mthd)
#         def wrapper(_iface:iGibsonSemanticInterface, *args:tuple, _sync_viewer:bool=True, **kwargs:Kwargs):
#             if _sync_viewer: 
#                 if _iface.has_gui:
#                     returnval = mthd(_iface, *args, **kwargs)
#                     _iface.env.simulator.sync()
#                     return returnval
#             return mthd(_iface, *args, **kwargs)
#         return wrapper


class MyiGibsonSemanticInterface(iGibsonSemanticInterface):

    @property
    def _pbclient(self):
        return self.env.simulator.cid

    ################################ Basic data/component access ################################
    @property
    def _robot(self) -> BaseRobot:
        assert len(self._robots)==1, \
            f"attribute '_robot' undefined for multi-robot {self.__class__.__name__} containing {len(self._robots)} robots: {self._robots}. "
        return self._robots[0]
    @property
    def robot(self) -> str:
        assert len(self.robots)==1, \
            f"attribute 'robot' undefined for multi-robot {self.__class__.__name__} containing robots: {self.robots}. "
        return self.robots[0]
    @property 
    def robot_bid(self) -> int:
        return self.get_bid(self._robot)
    @property
    def _eef(self) -> RobotLink:
        return self._robot.eef_links[self._robot.default_arm]
    @property
    def eef(self) -> str:
        return self._robot.eef_link_names[self._robot.default_arm]
    @property
    def eef_link(self) -> int:
        return self._eef.link_id 
    @property
    def _arm_joint_ids(self) -> List[int]:
        return self._motion_planner.arm_joint_ids
    @property
    def _arm_joint_control_indices(self) -> List[int]:
        return [*self._robot.trunk_control_idx, *self._robot.arm_control_idx[self._robot.default_arm]]
    
    # @_sync_viewer_after_exec
    def set_position(self, body:UID, position:Position) -> None:
        return super().set_position(body, position)

    # @_sync_viewer_after_exec
    def set_orientation(self, body:UID, orientation:Orientation, *, force_quaternion=True) -> None:
        return super().set_orientation(body, orientation, force_quaternion=force_quaternion)

    # @_sync_viewer_after_exec
    def set_position_orientation(self, body:UID, 
                                 position:Position, 
                                 orientation:Orientation, 
                                 *, 
                                 force_quaternion=True) -> None:
        return super().set_position_orientation(body=body, position=position, orientation=orientation, force_quaternion=force_quaternion)


    # @_sync_viewer_after_exec
    def set_joint_positions(self, body:UID, joints:Iterable[int], values:JointPos):
        return super().set_joint_positions(body, joints, values)


    ################################### Initialize Environment #####################################
    # TODO: generalize
    def __init__(self, config_file:str, objects:Collection[ObjectSpec]=[], *, headless:bool=True, tensor:bool=False, verbose=True):
        self.has_gui = not headless
        self._init_igibson(config=config_file, headless=headless, tensor=tensor)
        self.load_objects(objects, verbose=verbose)
        self._init_object_state(objects,verbose=verbose)
        self._init_robot_state()
        # Setup viewer if needed
        if self.has_gui:
            self._viewer_setup()

    def _init_igibson(self, config:Union[str,dict]={}, headless:bool=True, tensor:bool=False, **config_options:Kwargs) -> Tuple[iGibsonEnv, MotionPlanningWrapper]:
        # Load simulation config
        if isinstance(config,str): 
            config = self._load_config(config_file=config, **config_options)
        assert isinstance(config,dict)
        if not config['robot']['name'] == 'Fetch':
            raise NotImplementedError(f"{self.__class__.__name__} is only currently implemented for a single Fetch robot.")
        # Initialize iGibson simulation environment
        MODE = {
            (True,  True)  : "headless_tensor",
            (True,  False) : "headless",
            (False, True)  : "gui_tensor",
            (False, False) : "gui_interactive",
        }
        env = iGibsonEnv(
            config_file=config,
            mode=MODE[(headless,tensor)]
            # mode="gui_interactive" if not headless else "headless",
            # action_timestep=1.0 / 120.0,
            # physics_timestep=1.0 / 120.0,
        ) 
        self._env = env
        # Setup motion planning wrapper
        motion_planner = MotionPlanningWrapper(
            self.env,
            optimize_iter=10,
            full_observability_2d_planning=False,
            collision_with_pb_2d_planning=False,
            visualize_2d_planning=False,
            visualize_2d_result=False,
        )
        self._motion_planner = motion_planner
        
    
    def _load_config(self, config_file:str, load_object_categories:Optional[List[str]]=[], visualize_planning=False, **config_options:Kwargs) -> dict:
        config_file = os.path.abspath(config_file)
        config_data:dict = parse_config(config_file)
        
        # don't load default furniture, etc. to accelerate loading (only building structures)
        if load_object_categories is not None:
            config_data["load_object_categories"] = load_object_categories
        if not visualize_planning:
            config_data['visible_target'] = False
            config_data['visible_path'] = False
        config_data.update(config_options) # overwrite config file with specified options

        # Reduce texture scale for Mac.
        if sys.platform == "darwin":    config_data["texture_scale"] = 0.5
        # Save config data to private field 
        self._config = config_data
        return self._config
    

    
    def load_object(self, spec:Union[str,dict], verbose:bool=True) -> List[int] :
        URDF_kwargs = {
            'avg_obj_dims' : get_ig_avg_category_specs().get(spec["category"]),
            'fit_avg_dim_volume' : True,
            'texture_randomization' : False,
            'overwrite_inertial' : True,
        }
        
        spec = _as_urdf_spec(spec)
        URDF_kwargs.update(**spec)

        if verbose: print(f"Loading {spec['category'].capitalize()} object '{spec['name']}'...", end='')
        
        # Create and import the object
        sim_obj = URDFObject(**URDF_kwargs)
        self.env.simulator.import_object(sim_obj)
        
        if verbose: print(" done.")
                
        # Return obj body_ids
        return sim_obj.get_body_ids()

    def load_objects(self, specifications:Iterable[Union[str,dict]], verbose:bool=True) -> None:
        '''Load objects into simulator based on a list of categories.
        ''' 
        body_ids_by_object = {}
        for spec in specifications:
            spec = _as_urdf_spec(spec)
            body_ids = self.load_object(spec=spec, verbose=verbose)
            body_ids_by_object[spec["name"]] = body_ids 
        return body_ids_by_object
    
    def _init_object_state(self, obj_specs:Iterable[ObjectSpec], *, verbose=True) -> None:
        for spec in obj_specs:
            assert isinstance(spec, dict)
            if not isinstance(spec,ObjectSpec): 
                spec = ObjectSpec(**spec)
            for state in self._states:
                if state in spec:    
                    self.set_state(spec["name"], state, spec[state])

    def _init_robot_state(self, position:Position=(0,0,0), orientation=(0,0,0), tuck:bool=False) -> None:
        '''Initialize robot state. 
        
        WARNING: this steps sim; no objects can be added to scene after this is called.
        '''
        self.env.reset()
        self.env.land(self.env.robots[0], position, orientation) # note: this steps sim!
        for robot in self.env.robots:
            robot.tuck() if tuck else robot.untuck()
            robot.keep_still()
    
    # @_sync_viewer_after_exec
    def _viewer_setup(self, position:Position=[-0.8, 0.7, 1.7], orientation:Euler=[0.1, -0.9, -0.5], *, keep_viewer_on_top=True) -> None:
        # Set viewer camera 
        self.env.simulator.viewer.initial_pos = position
        self.env.simulator.viewer.initial_view_direction = orientation
        self.env.simulator.viewer.reset_viewer()

        # # This doesn't seem to be working
        # if keep_viewer_on_top:
        #     cv2.setWindowProperty("RobotView", cv2.WND_PROP_TOPMOST, 1)
        #     cv2.setWindowProperty("Viewer", cv2.WND_PROP_TOPMOST, 1)
    
    
    ################################################################################
    ################################################################################   

    _states = ("position", "orientation")
    _state_access_map = {
        "position" : (
            lambda self, body: self.get_position(body), 
            lambda self, body, val: self.set_position(body,val)
        ),
        "orientation" : (
            lambda self, body: self.get_orientation(body), 
            lambda self, body, val: ft.partial(self.set_orientation, force_quaternion=True)(body,val)
        ),
    }
    assert set(_states) == set(_state_access_map)

    def get_state(self, body:UID, state:str) -> Any:
        assert state in self._states, f"state '{state}' not registered in '_states' field of MyiGibsonSemanticInterface object."
        return MyiGibsonSemanticInterface._state_access_map[state][0](self,body)
    
    # @_sync_viewer_after_exec
    def set_state(self, body:UID, state:str, value:Any) -> None:
        assert state in self._states, f"state '{state}' not registered in '_states' field of MyiGibsonSemanticInterface object."
        MyiGibsonSemanticInterface._state_access_map[state][1](self, body, value)

    # ----------------------------------------------------------------------

    def get_pose(self, body:UID) -> Pose:
        try:
            return self.get_position_orientation(body)
        except KeyError as e:
            if body in ("eef", "gripper", self.eef_link):
                return self.get_eef_pose() # eef has to be queried differently
            else:
                raise e
    
    # @_sync_viewer_after_exec
    def set_pose(self, 
                 body:UID, 
                 pose:Optional[Pose]=None, 
                 *, 
                 position:Optional[Position]=None, 
                 orientation:Optional[Orientation]=None
                ) -> None:
        pose_specified = (pose is not None)
        pos_orn_specified = (position is not None) or (orientation is not None)
        assert(pose_specified ^ pos_orn_specified), \
            "must specify either 'pose' xor both 'position' and 'orientation'"
        pos, orn = pose if pose_specified else (position, orientation)
        self.set_position_orientation(body, pos, orn)

    def get_eef_pose(self) -> Pose:
        '''Queries sim calculate end-effector pose from current arm joint config using forward kinematics.
        ignored return vals in order are: 
          localInertialFramePosition: Position3D, localInertialFrameOrientation: Quaternion,
          worldLinkFramePosition: Position3D, worldLinkFrameOrientation: Quaternion
        In addition, if pybullet.getLinkState is called with computeLinkVelocity=1, there will be two additional return values:
          worldLinkLinearVelocity:Tuple[float,float,float], worldLinkAngularVelocity:Tuple[float,float,float]
          '''
        pos, orn, _, _, _, _ = pb.getLinkState(self.robot_bid, self.eef_link, computeForwardKinematics=True, physicsClientId=get_client())
        return (pos, orn)
    
    def get_arm_config(self) -> JointPos:
        return self.get_joint_positions(self.robot_bid, self._arm_joint_ids)
    
    # @_sync_viewer_after_exec
    def set_arm_config(self, q:JointPos, attachments:List[Attachment]=[], freeze=False) -> None:
        if isinstance(q,BodyConf): q = q.configuration
        self.set_joint_positions(self.robot_bid, self._arm_joint_ids, q)
        if freeze:
            self._robot.keep_still()

    # ----------------------------------------------------------------------
    
    def is_movable(self, body: UID):
        obj = self.get_object(body)
        return not obj.fixed_base

    def is_on(self, body:UID, surface:UID):
        body    = self.get_object(body)
        surface = self.get_object(surface)
        return body.states[object_states.OnTop].get_value(surface)
    
    ################################################################################

    def is_collision_free(self, 
                        body:UID, 
                        pose: Optional[Pose]=None,  # TODO: update to accept joint config
                        body2:Optional[UID]=None,
                        pose2:Optional[Pose]=None, 
    ) -> bool:
        assert((pose2 is None) or (body2 is not None)), "cannot specify arg 'pose2' without arg 'body2'"
    
        robot = self._robot
        with UndoableContext(robot):
            body = self.get_bid(body)
            body_links = self.get_object(body).get_body_ids()
            
            if body2 is not None:
                body2 = self.get_bid(body2)
                body2_links = self.get_object(body2).get_body_ids()
                if isinstance(body2,str): 
                    body2 = self._get_bodyids_from_name(body2)[0]
            else:
                body2 = body2_links = None

            if pose is not None:
                self.set_position_orientation(body, *pose)
            if pose2 is not None:
                self.set_position_orientation(body2, *pose2)
            
            return is_collision_free(body_a=body, 
                                     link_a_list=body_links,
                                     body_b=body2, 
                                     link_b_list=body2_links
                                    )
        
    def sample_arm_config(self, collisions=True, max_attempts=1000):
        robot = self._robot
        robot_bid = self.robot_bid
        joint_ids = self._arm_joint_ids
        sample_fn = get_sample_fn(robot_bid, joint_ids)
        collision_fn = self._get_robot_arm_collision_fn()
        with UndoableContext(robot):
            config = sample_fn()
            if collisions:
                for _ in range(max_attempts):
                    if not collision_fn(config):
                        return config
                return None
            else:
                return config

    def sample_placement(self, 
                         body:UID, 
                         surface:UID, 
                         ) -> Pose:
        robot = self._robot
        obj = self.get_object(body)

        with UndoableContext(robot):
            # pred_map = {object_states.OnTop: "onTop", object_states.Inside: "inside"}
            result = sample_kinematics(
                predicate="onTop",
                objA=obj,
                objB=self.get_object(surface),
                binary_state=True,
                use_ray_casting_method=True,
                max_trials=MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE,
                skip_falling=False,
                z_offset=PREDICATE_SAMPLING_Z_OFFSET,
            )

            if not result:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.SAMPLING_ERROR,
                    "Could not sample position with object and predicate.",
                    {f"object: {body}, target surface={surface}"},
                )

            pos, orn = obj.get_position_orientation()
            return pos, orn
    
    def get_collidable_body_ids(self, include_robot:Union[bool,UID,BaseRobot]=False):
        if include_robot is False: # could be the robot's UID
            disabled_bodies = self._robots
        elif include_robot is True:
            disabled_bodies = []
        else:
            included = [
                robot if isinstance(robot,BaseRobot) 
                else self.get_object(robot)
                for robot in nonstring_iterable(include_robot)
            ]
            disabled_bodies = [robot for robot in self._robots if not robot in included]

        obstacle_ids = [
            body_id
            for body_id in self._env.scene.get_body_ids()
            if  not any(body_id in floor.get_body_ids() for floor in self.env.scene.objects_by_category["floors"])
            and not any(body_id in  body.get_body_ids() for body  in disabled_bodies)
        ]

        return obstacle_ids   
    
    def get_collisions(self, 
                       body:UID, 
                       obstacles:Collection[UID]=[], 
                       *, 
                       exclude:Collection[UID]=[], 
                       critical_obstacles:Collection[UID]=[], 
                       max_distance:float=HAND_MAX_DISTANCE
                ) -> List[UID]:
        if not obstacles and not critical_obstacles:
            obstacles = [obj for obj in self.objects if not (obj==body or obj in exclude)]
        obstacles = set(obstacles) - set(exclude)

        close_objects = set(x[0] for x in pb.getOverlappingObjects(*get_aabb(body)))
        close_obstacles = (close_objects & set(obstacles)) | set(critical_obstacles)
        collision_ids = [obs for obs in close_obstacles if pairwise_collision(body, obs, max_distance=max_distance)]
        return collision_ids

    
    def get_robot_arm_collision_params(self,
                                *,
                                obstacles:List[int]=[],
                                ignore_other_scene_obstacles:bool=False,
                                self_collisions:bool=True, 
                                disabled_collisions:Set[Tuple[int,int]]=set(),
                                allow_collision_links:List[int]=[],
                                allow_gripper_collisions=False,
    ) -> Tuple[BaseRobot, int,List[int],List[int],Set[Tuple[int,int]],List[Tuple[int,int]],List[int],List[float],List[float]]:
        '''Assuming one Fetch robot so we can use MotionPlanningWrapper functionality
        '''
        robot     = self._robot
        robot_bid  = self.robot_bid
        joint_ids = self._arm_joint_ids
        if not ignore_other_scene_obstacles:
            obstacles = list(set(obstacles) | set(self._motion_planner.mp_obstacles))

        disabled_collisions = robot.disabled_collision_pairs + [collision for collision in disabled_collisions if collision not in robot.disabled_collision_pairs]
        if robot.model_name == "Fetch":
            # iGibson motion_planning_wrapper processing for Fetch
            fetch_disabled_collisions = [
                (link_from_name(robot_bid, "torso_lift_link"), link_from_name(robot_bid, "torso_fixed_link")),
                (link_from_name(robot_bid, "torso_lift_link"), link_from_name(robot_bid, "shoulder_lift_link")),
                (link_from_name(robot_bid, "torso_lift_link"), link_from_name(robot_bid, "upperarm_roll_link")),
                (link_from_name(robot_bid, "torso_lift_link"), link_from_name(robot_bid, "forearm_roll_link")),
                (link_from_name(robot_bid, "torso_lift_link"), link_from_name(robot_bid, "elbow_flex_link")),
            ]
            disabled_collisions += [collision for collision in fetch_disabled_collisions if collision not in disabled_collisions]
            
            if allow_gripper_collisions:
                allow_collision_links = set(allow_collision_links)
                allow_collision_links.add(self.eef_link)
                finger_links = set([finger.link_id for finger in robot.finger_links[robot.default_arm]])
                allow_collision_links |= finger_links
                allow_collision_links = list(allow_collision_links)

        
        # Pair of links within the robot that need to be checked for self-collisions
        self_collision_id_pairs = get_self_link_pairs(robot_bid, joint_ids, disabled_collisions) if self_collisions else []

        # List of links that move on the robot and that should be checked for collisions with the obstacles
        moving_links = frozenset([item for item in get_moving_links(robot_bid, joint_ids) if not item in allow_collision_links])
        moving_robot_bids = [(robot_bid, moving_links)]
        
        # Joint limits
        lower_limits, upper_limits = get_custom_limits(robot_bid, joint_ids)

        return robot, robot_bid, joint_ids, obstacles, disabled_collisions, allow_collision_links, self_collision_id_pairs, moving_robot_bids, lower_limits, upper_limits
    
    
    
    def arm_fk(self, q:JointPos):
        '''Arm Forward Kinematics: Calculate Fetch gripper pose given arm joint positions.
        '''
        with UndoableContext(self._robot):
            # ignored return vals in order are: 
            #   localInertialFramePosition: Position3D, localInertialFrameOrientation: Quaternion,
            #   worldLinkFramePosition: Position3D, worldLinkFrameOrientation: Quaternion
            # In addition, if pybullet.getLinkState is called with computeLinkVelocity=1, there will be two additional return values:
            #   worldLinkLinearVelocity:Tuple[float,float,float], worldLinkAngularVelocity:Tuple[float,float,float]
            self.set_arm_config(q) #, _sync_viewer=False)
            pos, orn, _, _, _, _ = pb.getLinkState(self.robot_bid, self.eef_link, computeForwardKinematics=True, physicsClientId=get_client())
            return (pos, orn)

    def arm_ik(self, eef_position:Position, eef_orientation:Optional[Orientation]=None, use_nullspace:bool=True, **kwargs) -> Optional[JointPos]:
        '''Arm Inverse Kinematics: Calculate Fetch arm joint configuration that satisfies the given end-effector position in the workspace (ignoring collisions). 
        Can additionally specify a target end-effector orientation and/or joint configuration nullspace.
        '''
        # with UndoableContext(self._robot):   
        *null_space, joint_damping = self._motion_planner.get_ik_parameters()
        threshold = self._motion_planner.arm_ik_threshold
        # ik_solver = pb.IK_DLS or pb.IK_SDLS
        ik_spec =  {
            # robot/eef body IDs
            'bodyUniqueId' : self.robot_bid, 
            'endEffectorLinkIndex' : self.eef_link, 
            # workspace target
            'targetPosition' : eef_position, 
            # IK solver args
            'jointDamping' : joint_damping,
            #'solver' : ik_solver,
            'maxNumIterations' : 100,
            'residualThreshold' : threshold,
            # additional pybullet.calculateInverseKinematics kwargs
            'physicsClientId' : get_client(), 
            **kwargs
        }
        
        # Determine if result should be constrained by a target orientation or the Fetch arm's nullspace
        orientation_spec = (
            {} if eef_orientation is None else                                          # no eef orientation specified
            {'targetOrientation': eef_orientation} if len(eef_orientation)==4 else 
            {'targetOrientation': quat_from_euler(eef_orientation)}        
        )
        nullspace_spec = (
            {} if not use_nullspace else
            dict(zip(['upperLimits', 'lowerLimits', 'restPoses', 'jointRanges'], null_space)) \
        )
        
        # Calculate IK
        q = pb.calculateInverseKinematics(**ik_spec, **orientation_spec, **nullspace_spec)

        # Check if config is valid
        if (q is None) or any(map(math.isnan, q)):
            return None
        
        # Extract joint angles from full configuration
        joint_vector_indices = self._motion_planner.robot_arm_indices
        q = tuple(np.array(q)[joint_vector_indices])

        return q
    
    def as_arm_config(self, kc:KinematicConstraint, disabled_types:Collection[KinematicConstraint]={}):
        disabled = lambda T: T in disabled_types
        if not disabled(BodyConf) and self._is_kinematic_constraint(kc,BodyConf):
            assert (kc.body == self.robot_bid),    f"BodyConf {kc} field 'body' does not match body ID of {self.robot}."
            assert (kc.joints == self._arm_joint_ids), f"BodyConf {kc} field 'joints' does not match arm joint IDs of {self.robot}."
            q = kc.configuration
        elif not disabled(BodyPose) and self._is_kinematic_constraint(kc,BodyPose):
            assert (kc.body==self.robot_bid) or (kc.body==self.eef_link) 
            q = self.arm_ik(*kc.pose)
        elif not disabled(Pose) and self._is_kinematic_constraint(kc,Pose):
            q = self.arm_ik(*kc)
        elif not disabled(JointPos) and self._is_kinematic_constraint(kc,JointPos):
            q = kc
        elif not disabled(Position) and self._is_kinematic_constraint(kc,Position):
            q = self.arm_ik(eef_position=kc)
        else:
            raise TypeError(
                f"Could not interpret input {kc} of type {type(kc)} as a joint configuration or end effector pose. '"
                f"Expected one of: JointPos, BodyConf, or EEF BodyPose, Position, or Pose. "
            )
        assert self._is_kinematic_constraint(q, JointPos)
        return q

    

    def _is_kinematic_constraint(self, x:Any, T:Type=KinematicConstraint) -> bool:
        '''May give false positives; tests aren't perfect.'''
        tests = {
            BodyConf :    (lambda x: isinstance(x,BodyConf)),
            BodyPose :    (lambda x: isinstance(x,BodyPose)),
            JointPos :    (lambda x: hasattr(x,"__len__") and len(x)==len(self._arm_joint_ids) and is_numeric_vector(x) ),
            Pose :        (lambda x: hasattr(x,"__len__") and len(x)==2 and is_numeric_vector(x[0])),
            Position :    (lambda x: hasattr(x,"__len__") and len(x)==3 and is_numeric_vector(x)),
        }
        if T==KinematicConstraint: # return true if x passes any KC test
            return any(t(x) for t in tests.values())
        elif T in tests:
            return tests[T](x)
        else:
            return ValueError(
                f"Type T={T} is not a KinematicConstraint type. Expected one of {[KinematicConstraint]+list(get_args(KinematicConstraint))}."
            )




    def _get_robot_arm_collision_fn(self,
                                *,
                                self_collisions:bool=True, 
                                disabled_collisions={},
                                allow_collision_links=[], 
                                allow_gripper_collisions=False,
                                **kwargs
    ):
        '''Assuming one Fetch robot so we can use MotionPlanningWrapper functionality

        Mimics iGibson pybullet_tools.utils.get_collision_fn
        '''
        ( 
            robot,
            robot_bid, 
            joint_ids, 
            obstacles, 
            disabled_collisions, 
            allow_collision_links,
            self_collision_id_pairs, 
            moving_robot_bids, 
            lower_limits, 
            upper_limits 
        ) \
        = self.get_robot_arm_collision_params(
            self_collisions=self_collisions, 
            disabled_collisions=disabled_collisions,
            allow_collision_links=allow_collision_links, 
            allow_gripper_collisions=allow_gripper_collisions,
        )

        def _ensure_joint_position_vector(
            qspace_collision_fn:Callable[[JointPos,Iterable[Attachment]],bool]
        ) -> Callable[[KinematicConstraint,Iterable[Attachment]],bool]:
            ft.wraps(qspace_collision_fn)
            def _wrapper(x:KinematicConstraint, attachments:Iterable[Attachment]=[]) -> bool :
                joint_positions = self.as_arm_config(x)
                return qspace_collision_fn(q=joint_positions, attachments=attachments)
            return _wrapper
                    
        @_ensure_joint_position_vector
        def arm_collision_fn(q:JointPos, attachments:Iterable[Attachment]=[]) -> bool:
            assert len(joint_ids) == len(q)
            
            with UndoableContext(robot):
                if not all_between(lower_limits, q, upper_limits):
                    return True
                self.set_arm_config(q) #, _sync_viewer=False)
                for attachment in attachments:
                    attachment.assign()
                ################################################################################
                # get_moving_aabb = cached_fn(get_buffered_aabb, cache=True, max_distance=max_distance/2., **kwargs) # pddlstream pybullet_tools.utils.get_collision_fn
                ################################################################################
                

                # Check for self collisions
                for link1, link2 in iter(self_collision_id_pairs):
                    if pairwise_link_collision(robot_bid, link1, robot_bid, link2):
                        return True
                    ################################################################################
                    ##### pddlstream pybullet_tools.utils.get_collision_fn:
                    # # Self-collisions should not have the max_distance parameter
                    # # TODO: self-collisions between body and attached_bodies (except for the link adjacent to the robot)
                    # if (not use_aabb or aabb_overlap(get_moving_aabb(body), get_moving_aabb(body))) and \
                    #         pairwise_link_collision(body, link1, body, link2): #, **kwargs):
                    #     #print(get_body_name(body), get_link_name(body, link1), get_link_name(body, link2))
                    #     if verbose: print(body, link1, body, link2)
                    #     return True
                    ################################################################################
                
                # Include attachments as "moving bodies" to be tested for collisions
                attached_bodies = [attachment.child for attachment in attachments]
                moving_body_ids = moving_robot_bids + attached_bodies

                # Check for collisions of the moving bodies and the obstacles
                for moving, obs in product(moving_body_ids, obstacles):
                    ################################################################################
                    ##### pddlstream pybullet_tools.utils.get_collision_fn:
                    # if (not use_aabb or aabb_overlap(get_moving_aabb(body1), get_obstacle_aabb(body2))) \
                    # and pairwise_collision(body1, body2, **kwargs):
                    ################################################################################
                    if pairwise_collision(moving, obs, **kwargs):
                        return True
                
                # No collisions detected
                return False

        return arm_collision_fn


    def _get_hand_collision_fn(self, max_distance:float=HAND_MAX_DISTANCE):
        robot:BaseRobot = self._robot
        eef:UID = self._eef.body_id
        non_hand_obstacles:Set[UID] = set(obs for obs in self.get_collidable_bodies(include_robot=True) if (obs != eef))
        get_close_objects = lambda body_id: set(x[0] for x in pb.getOverlappingObjects(*get_aabb(body_id)))
        
        def hand_collision_fn(pose3d, obj_in_hand:Optional[BaseObject]=None, *, report_all_collisions:bool=False, verbose:bool=False):
                          
            non_hand_non_oih_obstacles = set(
                obs
                for obs in non_hand_obstacles
                if (obj_in_hand is None) or (obs not in obj_in_hand.get_body_ids())
            )

            get_close_obstacles = lambda body_id: get_close_objects(body_id) & non_hand_non_oih_obstacles
            
            with UndoableContext(robot):
                robot.set_eef_position_orientation(*pose3d, robot.default_arm)
                
                collisions = self.get_collisions(
                    body=eef,
                    obstacles=non_hand_non_oih_obstacles, 
                    max_distance=max_distance
                )
                collision = (len(collisions)>0)
                if collision:
                    if verbose: 
                        print("EEF collision with objects: ", collisions)
                    if not report_all_collisions:
                        return collision # True

                if obj_in_hand is not None:
                    [oih_bid] = obj_in_hand.get_body_ids()
                    oih_collisions = self.get_collisions(
                        body=oih_bid,
                        obstacles=non_hand_non_oih_obstacles, 
                        critical_obstacles=get_close_obstacles(eef.body_id), 
                        max_distance=max_distance
                    )
                    oih_collision = (len(oih_collisions)>0)
                    if verbose and oih_collision:
                        print("Held object collision with objects: ", oih_collisions)
                    collision = collision or oih_collision

            return collision

        return hand_collision_fn
    


    
    
    def plan_arm_joint_motion(self, 
        q1:JointPos, 
        q2:JointPos, 
        obstacles:List[int]=[],
        attachments:List[Attachment]=[],
        weights:Optional[Iterable[float]]=None, 
        resolutions:Optional[Iterable[float]]=None, 
        algorithm:str='birrt',
        ignore_other_scene_obstacles=True,
        *, 
        planning_utils_version:PybulletToolsVersion=PybulletToolsVersion.IGIBSON, 
        use_aabb:bool=False, 
        cache:bool=True, 
         **kwargs
    ):
        '''My attempt to reconcile plan_joint_motion between the implementations used by pddlstream and iGibson.
        '''
        
        assert len(self._arm_joint_ids) == len(q1) and len(self._arm_joint_ids) == len(q2)

        distance_fn, sample_fn, extend_fn, collision_fn = self._get_arm_joint_motion_helper_fns(
            obstacles=obstacles,
            ignore_other_scene_obstacles=ignore_other_scene_obstacles,
            attachments=attachments,
            weights=weights,
            resolutions=resolutions,
            use_aabb=use_aabb, 
            cache=cache,
        )
        
        with UndoableContext(self._robot):
            if not pybullet_tools.utils.check_initial_end(q1, q2, collision_fn):
                return None
            
            mp_algo_fn = self._get_motion_planning_algorithm(algorithm)
            if algorithm == 'direct':
                return mp_algo_fn(q1, q2, extend_fn, collision_fn)
            elif algorithm == 'birrt':
                return mp_algo_fn(q1, q2, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs)
            elif algorithm == 'rrt_star':
                return mp_algo_fn(q1, q2, distance_fn, sample_fn, extend_fn, collision_fn, max_iterations=5000, **kwargs)
            elif algorithm == 'rrt':
                return mp_algo_fn(q1, q2, distance_fn, sample_fn, extend_fn, collision_fn, iterations=500, **kwargs)
            elif algorithm == 'lazy_prm':
                return mp_algo_fn(q1, q2, distance_fn, sample_fn, extend_fn, collision_fn, [500, 2000, 5000], **kwargs)
            else:
                return None
    
    def _get_motion_planning_algorithm(self, algorithm):
        if algorithm == 'direct':
            return import_from('rrt_connect', targets=['direct_path'], package=motion) 
        elif algorithm == 'birrt':
            return import_from('rrt_connect', targets=['birrt'], package=motion) 
        elif algorithm == 'rrt_star':
            return import_from('rrt_star', targets=['rrt_star'], package=motion) 
        elif algorithm == 'rrt':
            return import_from('rrt', targets=['rrt'], package=motion) 
        elif algorithm == 'lazy_prm':
            return import_from('lazy_prm', targets=['lazy_prm_replan_loop'], package=motion) 
        else:
            raise ValueError(f"Inappropriate argument algorithm={algorithm}. Expected one of: 'direct', 'birrt', 'rrt_star', 'rrt', or 'lazy_prm'")


    def _get_arm_joint_motion_helper_fns(self, 
        obstacles=[],
        attachments=[], 
        self_collisions=True, 
        disabled_collisions=set(),
        weights=None, 
        resolutions=None,
        ignore_other_scene_obstacles=False,
        *, 
        use_aabb=False, 
        cache=True, # original pybullet_tools version only
        allow_collision_links=[], # iGibson pybullet_tools version only
        allow_gripper_collisions=False, # iGibson pybullet_tools version only
    ):
        ( 
            _,
            robot_bid, 
            joint_ids, 
            obstacles, 
            disabled_collisions, 
            allow_collision_links,
            self_collisions, 
            _, 
            _, 
            _ 
        ) \
        = self.get_robot_arm_collision_params(
            self_collisions=self_collisions, 
            disabled_collisions=disabled_collisions,
            allow_collision_links=allow_collision_links, 
            allow_gripper_collisions=allow_gripper_collisions,
        )
        # = self.get_robot_arm_collision_params(
        #     self_collisions=self_collisions, 
        #     disabled_collisions=disabled_collisions,
        #     ignore_other_scene_obstacles=ignore_other_scene_obstacles
        # )

        distance_fn = pybullet_tools.utils.get_distance_fn(robot_bid, joint_ids, weights=weights)
        sample_fn = pybullet_tools.utils.get_sample_fn(robot_bid, joint_ids)
        extend_fn = pybullet_tools.utils.get_extend_fn(robot_bid, joint_ids, resolutions=resolutions)
        
        assert isinstance(VERSION, PybulletToolsVersion)
        if VERSION==PybulletToolsVersion.PDDLSTREAM:
            if (weights is None) and (resolutions is not None):
                weights = np.reciprocal(resolutions)
            collision_fn = pybullet_tools.utils.get_collision_fn(
                body=robot_bid, joints=joint_ids, obstacles=obstacles, attachments=attachments, 
                self_collisions=self_collisions, disabled_collisions=disabled_collisions,
                use_aabb=use_aabb, cache=cache
            )
        elif VERSION==PybulletToolsVersion.IGIBSON:
            collision_fn = pybullet_tools.utils.get_collision_fn(
                body=robot_bid, joints=joint_ids, obstacles=obstacles, attachments=attachments, 
                self_collisions=self_collisions, disabled_collisions=disabled_collisions,
                allow_collision_links=allow_collision_links
            )

        return distance_fn, sample_fn, extend_fn, collision_fn


    


    def get_grasp_poses_for_object(self, target:BID, force_allow_any_extent=True):
        robot = self._robot
        target_obj = self.get_object(target)

        GUARANTEED_GRASPABLE_WIDTH = 0.1
        GRASP_OFFSET = np.array([0, 0, -0.08])  # 5cm back and 8cm up.
        # GRASP_OFFSET = np.array([0, 0.05, -0.08])  # 5cm back and 8cm up.
        GRASP_ANGLE = 0.35  # 20 degrees
        NUM_LATERAL_SAMPLES = 10
        LATERAL_MARGIN = 0.05

        bbox_center_in_world, bbox_quat_in_world, bbox_extent_in_base_frame, _ = target_obj.get_base_aligned_bounding_box(
            visual=False
        )

        # Normally just try to grasp the small-enough dimensions.
        allow_any_extent = True
        if not force_allow_any_extent and np.any(bbox_extent_in_base_frame < GUARANTEED_GRASPABLE_WIDTH):
            allow_any_extent = False

        grasp_candidates = []
        all_possible_idxes = set(range(3))
        for pinched_axis_idx in all_possible_idxes:
            # If we're only accepting correctly-sized sides, let's check that.
            if not allow_any_extent and bbox_extent_in_base_frame[pinched_axis_idx] > GUARANTEED_GRASPABLE_WIDTH:
                continue

            # For this side, grab all the possible candidate positions (the other axes).
            other_axis_idxes = all_possible_idxes - {pinched_axis_idx}

            # For each axis we can go:
            for palm_normal_axis_idx in other_axis_idxes:
                positive_option_for_palm_normal = np.eye(3)[palm_normal_axis_idx]

                for possible_side in [positive_option_for_palm_normal, -positive_option_for_palm_normal]:
                    grasp_center_pos = pb.multiplyTransforms(
                        bbox_center_in_world,
                        bbox_quat_in_world,
                        possible_side * bbox_extent_in_base_frame / 2,
                        [0, 0, 0, 1],
                    )[0]
                    towards_object_in_world_frame = bbox_center_in_world - grasp_center_pos
                    towards_object_in_world_frame /= np.linalg.norm(towards_object_in_world_frame)

                    # For the rotation, we want to get an orientation that points the palm towards the object.
                    # The palm normally points downwards in its default position. We fit a rotation to this transform.
                    palm_normal_in_bbox_frame = -possible_side

                    # We want the hand's backwards axis, e.g. the wrist, (which is Y+) to face closer to the robot.
                    possible_backward_vectors = []
                    for facing_side in [-1, 1]:
                        possible_backward_vectors.append(np.eye(3)[pinched_axis_idx] * facing_side)

                    possible_backward_vectors_in_world = np.array(
                        [
                            pb.multiplyTransforms(grasp_center_pos, bbox_quat_in_world, v, [0, 0, 0, 1])[0]
                            for v in possible_backward_vectors
                        ]
                    )

                    backward_vector_distances_to_robot = np.linalg.norm(
                        possible_backward_vectors_in_world - np.array(robot.get_position())[None, :], axis=1
                    )
                    best_wrist_side_in_bbox_frame = possible_backward_vectors[np.argmin(backward_vector_distances_to_robot)]

                    # Fit a rotation to the vectors.
                    lateral_in_bbox_frame = np.cross(best_wrist_side_in_bbox_frame, palm_normal_in_bbox_frame)
                    flat_grasp_rot = get_hand_rotation_from_axes(
                        lateral_in_bbox_frame, best_wrist_side_in_bbox_frame, palm_normal_in_bbox_frame
                    )

                    # Finally apply our predetermined rotation around the X axis.
                    bbox_orn_in_world = Rotation.from_quat(bbox_quat_in_world)
                    unrotated_grasp_orn_in_world_frame = bbox_orn_in_world * flat_grasp_rot
                    grasp_orn_in_world_frame = unrotated_grasp_orn_in_world_frame * Rotation.from_euler("X", -GRASP_ANGLE)
                    grasp_quat = grasp_orn_in_world_frame.as_quat()

                    # Now apply the grasp offset.
                    grasp_pos = grasp_center_pos + unrotated_grasp_orn_in_world_frame.apply(GRASP_OFFSET)

                    # Append the center grasp to our list of candidates.
                    grasp_pose = (grasp_pos, grasp_quat)
                    grasp_candidates.append((grasp_pose, towards_object_in_world_frame))

                    # Finally, we want to sample some different grasp candidates that involve lateral movements from
                    # the center grasp.
                    # First, compute the range of values we can sample from.
                    lateral_axis_idx = (other_axis_idxes - {palm_normal_axis_idx}).pop()
                    half_extent_in_lateral_direction = bbox_extent_in_base_frame[lateral_axis_idx] / 2
                    lateral_range = half_extent_in_lateral_direction - LATERAL_MARGIN
                    if lateral_range <= 0:
                        continue

                    # If there is a valid range, grab some samples from there.
                    lateral_distance_samples = np.random.uniform(-lateral_range, lateral_range, NUM_LATERAL_SAMPLES)
                    lateral_in_world_frame = bbox_orn_in_world.apply(lateral_in_bbox_frame)
                    for lateral_distance in lateral_distance_samples:
                        offset_grasp_pos = grasp_pos + lateral_distance * lateral_in_world_frame
                        offset_grasp_pose = (offset_grasp_pos, grasp_quat)
                        grasp_candidates.append((offset_grasp_pose, towards_object_in_world_frame))

        return grasp_candidates

    

    ################################################################################
    
    def get_stable_gen(self): # -> Generator[Pose, None, None]:
        def gen(body:UID, surface:UID):
            while True:
                try:
                    placement = self.sample_placement(body, surface)
                except ActionPrimitiveError:
                    continue
                if placement is not None:
                    pose = BodyPose(body, placement)
                    yield (pose,)
        return gen
    
    

    def get_grasp_gen(self):        
        def grasps(body:UID):
            target = self.get_bid(body)
            grasp_poses:List[Tuple[BodyPose,Orientation]] = self.get_grasp_poses_for_object(target=target, force_allow_any_extent=False)
            for grasp_pose, target_orn in grasp_poses:
                approach_position = grasp_pose[0] + target_orn * GRASP_APPROACH_DISTANCE
                approach_pose = (approach_position, grasp_pose[1])
                body_grasp = BodyGrasp(
                    body=target, 
                    grasp_pose=grasp_pose,
                    approach_pose=approach_pose,
                    robot=self.robot_bid,
                    link=self.eef_link
                )
                yield (body_grasp,)
        return grasps


    # def get_ik_fn(self, fixed:Optional[List[UID]]=None):
    #     robot = self._get_bodyids_from_name(self.robot)[0]
    #     obstacles = [self._get_bodyids_from_name(f)[0] for f in fixed] \
    #                 if fixed is not None \
    #                 else self.get_collidable_body_ids()
    #     print('ROBOT    \t\t', robot)
    #     print('OBSTACLES\t\t', obstacles)
    #     print()
    #     _fn = get_ik_fn(robot, obstacles)
    #     def fn(body:str, grasp:BodyGrasp):
    #         body = self._get_bodyids_from_name(body)[0]
    #         pose = BodyPose(body, self.get_pose(body))
    #         print('BODY    \t\t',body)
    #         print('POSE    \t\t',[f"{self._fmt_num_iter(elem)}  " for elem in pose.pose])
    #         x,y,z = grasp.grasp_pose[0]
    #         a,b,c,d = grasp.grasp_pose[1]
    #         u,v,w = grasp.approach_pose[0]
    #         p,q,r,s = grasp.approach_pose[1]
    #         grasp_body = self.get_name(grasp.body)
    #         grasp_robot = self.get_name(grasp.robot)
    #         # R = self._get_object_from_bodyid(grasp.robot)
    #         print(
    #             f"\nBody Grasp <{grasp}>:"
    #             f"\n  - Body:           name: {grasp_body}\tBID: {grasp.body}"
    #             f"\n  - Grasp Pose:     Position: ({x:.2f},{y:.2f},{z:.2f})    \tOrientation: ({a:.2f},{b:.2f},{c:.2f},{d:.2f}) "
    #             f"\n  - Approach pose:  Position: ({u:.2f},{v:.2f},{w:.2f})    \tOrientation: ({p:.2f},{q:.2f},{r:.2f},{s:.2f}) "
    #             f"\n  - Robot:          name: {grasp_robot}\tBID: {grasp.robot}"
    #             f"\n  - Link:           {grasp.link}"
    #             f"\n  - Index:          {grasp.index}"
    #         )
    #         print()
    #         return _fn(body, pose, grasp)
    #     return fn


    def get_grasp_traj_fn(self, num_attempts:int=10) -> Callable[[str,BodyGrasp],Tuple[JointPos,Command]]:
        joint_ids = self._arm_joint_ids
        max_limits, min_limits, rest_position, joint_range, joint_damping = self._motion_planner.get_ik_parameters()
        
        def calculate_grasp_command(target:UID, grasp:BodyGrasp) -> Tuple[JointPos,Command]:
            '''Calculate the actions needed to grasp the target object.
            
            Returns a tuple of a nearby 'approach' position for initiating the grasp, and
            a Command object containing a BodyPath from the approach to the grasp configuration, 
            an Attach action, and then the reversed BodyPath back to the approach pose; 
            or None if no valid path could be found in the number of attempts specified.

            :param target: UID, name or body_id of target object to grasp
            :param grasp: BodyGrasp, grasp info for environment robot grasping target object with poses in world frame. 
            '''
            with UndoableContext(self._robot):
                target_bid = self.get_bid(target)

                # Some sanity checks
                assert isinstance(grasp, BodyGrasp), f"grasp must be defined as a BodyGrasp object, not {type(grasp)}"
                assert grasp.robot == self.robot_bid, f"grasp must be defined for robot body ID {self.robot_bid}, not {grasp.robot}" 
                assert grasp.link == self.eef_link, f"grasp must be defined for eef body ID {self.eef_link}, not {grasp.link}" 
                assert grasp.body == target_bid, \
                    f"grasp must be defined for object body ID {target_bid} ({self.get_name(target_bid)}), not {grasp.body} ({self.get_name(grasp.body)})"

                ### Get joint angles for desired approach and grasp poses
                approach_position, _ = grasp.approach_pose
                approach_config = self.arm_ik(eef_position=approach_position)
                grasp_position, grasp_orientation = grasp.grasp_pose
                grasp_config = self.arm_ik(eef_position=grasp_position, eef_orientation=grasp_orientation)
                
                # Find a collision-free path from approach to grasp config
                for _ in range(num_attempts):
                    self.set_arm_config(approach_config) # , _sync_viewer=False)
                    path = self._motion_planner.plan_arm_motion(arm_joint_positions=grasp_config)
                    if path is not None:
                        # Starting at approach config, move to grasp config, perform grasp, and return to approach config
                        command = Command([ BodyPath(self.robot_bid, path),
                                            Attach(target_bid, self.robot_bid, self.eef_link),
                                            BodyPath(self.robot_bid, path[::-1], attachments=[grasp])])
                        return (approach_config, command)
        return calculate_grasp_command

    def _fmt_num_iter(self, q: JointPos, dec=3, iterable_type=tuple) -> None:
        return f"{iterable_type([round(qq,dec) for qq in q])}"

    def print_config(self, conf:Union[BodyConf,JointPos]) -> None:
        if isinstance(conf,BodyConf):
            print(
            f"BodyConf <{conf}>:"
            f"\n  - Body:           name: {self.get_name(conf.body)}\tBID: {conf.body}"
            f"\n  - Configuration:  {self._fmt_num_iter(conf.configuration)}"
            f"\n  - Index:          {conf.index}"
        )
        else:
            print(f"Configuration:   {self._fmt_num_iter(conf)}")
    def print_path(self, path:Iterable[JointPos]):
        print(
            f"Path:    \t{[self._fmt_config(q,2) for q in path]}"
        )



    
    def assign_poses(self, fluents:Iterable[Tuple[UID,...]]) -> Set[int]:
        obstacles = set()
        for fluent in fluents:
            name, *args = fluent
            if name.lower() == 'atpose':
                obj, pose = args
                obj = self.get_bid(obj)
                assert isinstance(obj, int)
                if self._is_kinematic_constraint(pose,BodyPose):
                    assert pose.body == obj
                    pose = pose.pose
                assert self._is_kinematic_constraint(pose,Pose)
                obstacles.add(obj)
                self.set_pose(obj, pose)
            else:
                raise ValueError(name)
        return obstacles


    def get_free_motion_gen(self):
        robot = self._robot
        robot_bid = self.robot_bid
        joint_ids = self._arm_joint_ids
        def qfree_traj(q1:JointPos, q2:JointPos, atpose_fluents:List[Tuple]=[]) -> Optional[Command]:
            with UndoableContext(robot):
                # q1.assign()
                self.set_arm_config(q1) #, _sync_viewer=False)
                # Temporarily use only specified obstacles for motion planner
                original_mp_obstacles = self._motion_planner.mp_obstacles
                self._motion_planner.mp_obstacles = set(self.get_collidable_body_ids()) | self.assign_poses(atpose_fluents)
                # Calculate arm trajectory
                path = self._motion_planner.plan_arm_motion(q2)
                # Reassign old mp_obstacles to original value
                self._motion_planner.mp_obstacles = original_mp_obstacles
                # Return value
                if path is not None:
                    command = Command([BodyPath(robot_bid, path, joints=joint_ids)]) if path is not None else None
                    return (command,)
                else:
                    return tuple()

        def _consistency_check(q:KinematicConstraint) -> None:
            if isinstance(q,BodyConf):
                assert (q.body == robot_bid) and (q.joints == joint_ids)
            elif isinstance(q, BodyPose):
                assert (q.body == robot_bid)

        ft.wraps(qfree_traj)
        def wrapper(q1:KinematicConstraint, q2:KinematicConstraint, atpose_fluents:List[Tuple]=[]) -> Optional[Command]:
            _consistency_check(q1)
            _consistency_check(q2)
            q1 = self.as_arm_config(q1)
            q2 = self.as_arm_config(q2)
            return qfree_traj(q1, q2, atpose_fluents=atpose_fluents)
        return wrapper


    def get_motion_gen(self):
        robot_bid = self.robot_bid
        joint_ids = self._arm_joint_ids
        def qfree_traj(q1:JointPos, q2:JointPos, grasp:BodyGrasp=None, atpose_fluents:List[Tuple]=[]) -> Optional[Command]:
            obstacles = set(self.get_collidable_body_ids()) | self.assign_poses(atpose_fluents)
            path = self.plan_arm_joint_motion(q1=q1, q2=q2, obstacles=obstacles, attachments=[grasp.attachment()])
            print(f"Path: {path}")
            if path is not None:
                command = Command([BodyPath(robot_bid, path, joints=joint_ids)])
                return (command,)
            else:
                return tuple()
                
        def _consistency_check(q:KinematicConstraint) -> None:
            if isinstance(q,BodyConf):
                assert (q.body == robot_bid) and (q.joints == joint_ids)
            elif isinstance(q, BodyPose):
                assert (q.body == robot_bid)

        ft.wraps(qfree_traj)
        def wrapper(q1:KinematicConstraint, q2:KinematicConstraint, grasp:BodyGrasp=None, atpose_fluents:List[Tuple]=[]) -> Optional[Command]:
            _consistency_check(q1)
            _consistency_check(q2)
            q1 = self.as_arm_config(q1)
            q2 = self.as_arm_config(q2)
            return qfree_traj(q1, q2, grasp=grasp, atpose_fluents=atpose_fluents)
        return wrapper 




    def get_cfree_pose_pose_test(self):
        return self.is_collision_free
    
    def get_cfree_approach_obj_pose_test(self):
        def collision_test(b1:UID, p1:Pose, g1:BodyGrasp, b2:UID, p2:Pose) -> bool:
            '''Determine if, when performing grasp g1 on target object b1 with pose p1,
            retracting back from grasp.grasp_pose to grasp.approach_pose will cause b1
            to collide with obstacle object b2 with pose p2.
            '''
            if (b1 == b2): 
                return False
            with UndoableContext(self._robot):
                self.set_pose(b2, p2) #, _sync_viewer=False)
                for obj_pose in interpolate_poses(g1.grasp_pose, g1.approach_pose):
                    collision_free = self.is_collision_free(b1,obj_pose,b2,p2)
                    if not collision_free:
                        return True
                    # self.set_pose(b1, obj_pose)
                    # if pairwise_collision(b1, b2):
                    #     return True
                return False
        
        def _preprocess_inputs(b1:UID, p1:Union[Pose,BodyPose], g1:BodyGrasp, 
                               b2:UID, p2:Union[Pose,BodyPose]
                            ) -> Tuple[int,Pose,BodyGrasp,int,Pose]:
            if not isinstance(b1, int): 
                b1 = self.get_bid(b1)
            if not isinstance(b2, int): 
                b2 = self.get_bid(b2)
            if isinstance(p1, BodyPose):
                assert p1.body == b1
                p1 = p1.pose
            if isinstance(p2, BodyPose):
                assert p2.body == b2
                p2 = p2.pose
            assert g1.body == b1
            return b1,p1,g1,b2,p2

        def collision_free_grasp_retrieval_test(b1:UID, p1:Union[Pose,BodyPose], g1:BodyGrasp, 
                                                b2:UID, p2:Union[Pose,BodyPose]) -> bool:
            '''Returns true if straight line trajectory of body b1 located initially at pose p1 caused by the 
            retrieval motion from g1.grasp_pose to g1.approach_pose is COLLISION-FREE w.r.t. the single obstacle 
            b2 located at pose p2.
            '''
            b1,p1,g1,b2,p2 = _preprocess_inputs(b1,p1,g1,b2,p2) 
            collision = collision_test(b1,p1,g1,b2,p2) 
            return not collision
       
        return collision_free_grasp_retrieval_test

    
    
    def get_command_obj_pose_collision_test(self):
        def collision_test(command:Command, body:int, pose:Pose) -> bool:
            '''Determine if the motions encoded by 'command' cause a collision with
            obstacle 'body' located at 'pose'.
            '''
            def _path_collision_test(path:BodyPath) -> bool:
                '''Determine if the moving bodies of 'path' collide with 'body' located at 'pose'.
                '''
                moving = path.bodies()
                if body in moving:
                    # Assume that moving bodies are not considered obstacles and cannot cause collisions
                    return False
                with UndoableContext(self._robot):
                    for _ in path.iterator():
                        if any(pairwise_collision(mov,body) for mov in moving):
                            return True
                    return False
            
            collision = any(_path_collision_test(path) for path in command.body_paths)
            return collision
        
        def _preprocess_inputs(command:Command, body:UID, pose:Union[Pose,BodyPose]) -> Tuple[Command,int,Pose]:
            if not isinstance(body,int): body = self.get_bid(body)
            if isinstance(pose,BodyPose):
                assert pose.body == body
                pose = pose.pose
            return command, body, pose
        
        def test(command:Command, body:UID, pose:Union[Pose,BodyPose]) -> bool:
            '''Returns true if the moving bodies of 'command' DO NOT COLLIDE with 'body' located at pose 'pose'.
            '''
            command, body, pose = _preprocess_inputs(command, body, pose)
            collision = collision_test(command, body, pose)
            return collision
        
        return test

    ################################################################################

class LockRenderer(Saver):
    '''disabling rendering temporarily makes adding objects faster
    '''
    def __init__(self, ig:MyiGibsonSemanticInterface, lock:bool=True):
        self.ig = ig
        self.client = ig.env.simulator.cid
        self.originally_locked = not lock
        self.set_lock(lock=lock)

    def _ignore_if_headless(mthd):
        @ft.wraps(mthd)
        def _wrapper(self, *args, **kwargs):
            # only need to do something if rendering is active
            if self.has_gui():
                return mthd(self, *args, **kwargs)
        return _wrapper

    @_ignore_if_headless
    def set_lock(self, lock=True):
        self.locked = lock
        self._set_renderer(enable=(not lock))
    
    @_ignore_if_headless
    def restore(self):
        if self.locked != self.originally_locked:
            self.set_lock(lock=self.originally_locked)

    def has_gui(self):
        GUI_MODES = (SimulatorMode.GUI_INTERACTIVE, SimulatorMode.GUI_NON_INTERACTIVE)
        return self.ig.env.simulator.mode in GUI_MODES
    
    @_ignore_if_headless
    def _set_renderer(self, enable):
        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, int(enable), physicsClientId=self.ig._pbclient)

    
################################################################################

def format_config(conf:Union[BodyConf,JointPos], dec=3):
    if isinstance(conf,BodyConf):
        conf = conf.configuration
    conf = round_numeric(conf, dec)
    return conf



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g','--gui', action='store_true', help='Simulates the system')
    args = parser.parse_args()
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = "fetch_tamp.yaml"
    config_path = os.path.join(dir_path,config)

    sim = MyiGibsonSemanticInterface(
        config_file=config_path, 
        headless=(not args.gui)
    )

    def test_fn(nonterminal):
        return np.cumsum(nonterminal, 0) if isinstance(nonterminal, np.ndarray) else nonterminal

    # x = sim.get_arm_config()
    # config_pprint_str(x)
    x = sim.get_pose(sim.robot_bid) #[0][0]
    print(x)
    processed = recursive_map_advanced(
        ft.partial(round,ndigits=5), 
        x, 
        nonterminal_pre_recursion_fn=test_fn,
        nonterminal_post_recursion_fn=tuple,
        preserve_iterable_types=False
    )
    print(processed)
    # print(round_numeric(x))
    # print(truncate_floats(x))



    
if __name__=="__main__":
    main()
    
