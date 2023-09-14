from collections import UserDict
from enum import Enum
import functools as ft
from itertools import product
import math
import os
import sys
from typing import Any, Callable, Collection, Set, Iterator, Iterable, List, Optional, Tuple, Type, TypeVar, Union
from typing_extensions import TypeAlias

import pybullet as pb
import numpy as np

from igibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
from igibson.action_primitives.starter_semantic_action_primitives import GRASP_APPROACH_DISTANCE, MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE, PREDICATE_SAMPLING_Z_OFFSET, UndoableContext
from igibson import object_states
from igibson.external.pybullet_tools.utils import all_between, create_attachment, get_aabb, get_collision_fn, get_custom_limits, get_moving_links, get_self_link_pairs, is_collision_free, link_from_name, pairwise_collision, pairwise_link_collision, set_joint_positions
from igibson.object_states.utils import sample_kinematics
from igibson.objects.object_base import BaseObject
from igibson.envs.igibson_env import iGibsonEnv
from igibson.robots import BaseRobot
from igibson.robots.robot_base import RobotLink
from igibson.scenes.igibson_indoor_scene import URDFObject
from igibson.utils.assets_utils import get_ig_avg_category_specs
from igibson.utils.behavior_robot_motion_planning_utils import HAND_MAX_DISTANCE
from igibson.utils.grasp_planning_utils import get_grasp_poses_for_object
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
from igibson.utils.utils import parse_config, quatToXYZW
from transforms3d.euler import euler2quat

from examples.fetch.from_kuka_tamp.object_spec import (
    Euler,
    ObjectSpec,
    Orientation3D,
    Position3D,
    Quaternion, 
    URDFObjectSpec,
    Orientation, 
    Position,
    UniqueID, 
)
from examples.pybullet.utils.motion.motion_planners.rrt_connect import birrt
from examples.pybullet.utils.pybullet_tools.kuka_primitives import (
    Attach,
    BodyConf,
    BodyGrasp,
    BodyPath, 
    BodyPose,
    Command,
    get_grasp_gen,
    get_ik_fn, 
)
from examples.pybullet.utils.pybullet_tools.utils import MAX_DISTANCE, Attachment, check_initial_end, get_client, get_distance_fn, get_extend_fn, get_joint_positions, get_sample_fn, interpolate_poses, inverse_kinematics_helper, plan_direct_joint_motion, plan_joint_motion, plan_waypoints_joint_motion, quat_from_euler

# Pybullet Planning implementation: original by Caelan Reed Garrett vs. iGibson version 
PlanningUtilsVersion = Enum('PBToolsImpl', ['PDDLSTREAM', 'IGIBSON'])
import examples.pybullet.utils.pybullet_tools as pbtools_ps
import igibson.external.pybullet_tools as pbtools_ig

import examples.pybullet.utils.motion.motion_planners as motion_ps
import igibson.external.motion.motion_planners as motion_ig
PYBULLET_TOOLS_MODULES = {
    PlanningUtilsVersion.PDDLSTREAM : pbtools_ps,
    PlanningUtilsVersion.IGIBSON : pbtools_ig
}
MOTION_PLANNING_MODULES = {
    PlanningUtilsVersion.PDDLSTREAM : motion_ps,
    PlanningUtilsVersion.IGIBSON : motion_ig
}



# Opt: TypeAlias = Optional[Any]
# T = TypeVar('T')

JointPos: TypeAlias = Iterable[float]
Pose: TypeAlias = Tuple[Position,Orientation]
# Arm kinematic specification/constraint
KinematicConstraint: TypeAlias = Union[BodyConf,BodyPose,JointPos,Pose,Position,Orientation]



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

def elements_satisfy(iterable:Iterable, condition:Callable[[Any],bool])->bool:
    return all(condition(elem) for elem in iterable)

def is_iterable_with(
        iterable:Iterable, 
        condition:Optional[Callable[[Iterable],bool]]=lambda it: True, 
        element_condition:Optional[Callable[[Any],bool]]=None
) -> bool:
    return (
        # confirm that iterable is indeed an Iterable
        hasattr(iterable, "__iter__") and               
        # if condition was specified, confirm that iterable satisfies it
        (condition is None or condition(iterable)) and  
         # if element_condition was specified, confirm that every element in iterable satisfies it
        (element_condition is None or all(element_condition(elem) for elem in iterable))   
    )


def is_numeric(x:Any):
    return isinstance(x,(int,float))

def is_numeric_vector(v:Any):
    # return hasattr(v, "__iter__") and all(is_numeric(x) for x in v)
    return is_iterable_with(v, element_condition=is_numeric)

def is_nonstring_iterable(x:Any) -> bool:
    return hasattr(x, "__iter__") and not isinstance(x, str)
    
def nonstring_iterable(v:Any) -> Iterator:
    if is_nonstring_iterable(v): return iter(v)
    elif isinstance(v,str): return iter([v])
    else: return iter(v)







class iGibsonSemanticInterface:

    def __init__(self, env:iGibsonEnv):
        self._env = env

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

    # Object querying
    def get_id(self, obj:Union[UniqueID,BaseObject]) -> int:
        if isinstance(obj,BaseObject): return obj.get_body_ids()[0]
        elif isinstance(obj,str): return self._get_bodyids_from_name(obj)[0]
        elif isinstance(obj,int):
            assert obj in self.env.scene.objects_by_id, f"invalid body_id {obj}"
            return obj
        else:
            raise TypeError(
                f"Inappropriate argument obj={obj} of type {type(obj)}. " 
                "Expected BaseObject, str, or int"
            )
    def get_name(self, obj:Union[UniqueID,BaseObject]) -> BaseObject:
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
    def get_object(self, obj:Union[UniqueID,BaseObject]) -> BaseObject:
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
        

    # Position getter/setter
    def get_position(self, body:UniqueID) -> Position:
        return self.get_object(body).get_position()
    def set_position(self, body:UniqueID, position:Position) -> None:
        self.get_object(body).set_position(pos=position)

    # Orientation getter/setter
    def get_orientation(self, body:UniqueID) -> Orientation:
        return self.get_object(body).get_orientation()    
    def set_orientation(self, body:UniqueID, orientation:Orientation, *, force_quaternion=False) -> None:
        if force_quaternion and len(orientation)==3:
            orientation = quatToXYZW(euler2quat(*orientation), "wxyz")
        self.get_object(body).set_orientation(orn=orientation)

    # Combined Position and Orientation getter/setter
    def get_position_orientation(self, body:UniqueID) -> Pose:
        return self.get_object(body).get_position_orientation()
    def set_position_orientation(self, body:UniqueID, 
                                 position:Position, 
                                 orientation:Orientation, 
                                 *, 
                                 force_quaternion=False) -> None:
        self.set_position(body=body, position=position)
        self.set_orientation(body=body, orientation=orientation, force_quaternion=force_quaternion)

    # Joint states
    def get_joint_states(self, body:UniqueID):
        return self.get_object(body).get_joint_states()


    

class MyiGibsonSemanticInterface(iGibsonSemanticInterface):


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
    def robot_id(self) -> int:
        return self.get_id(self._robot)

    @property
    def _eef(self) -> RobotLink:
        return self._robot.eef_links[self._robot.default_arm]
        # return self._robot._links[self.eef]
    @property
    def eef(self) -> str:
        return self._robot.eef_link_names[self._robot.default_arm]
    @property
    def eef_id(self) -> int:
        return self._eef.link_id 
    
    @property
    def _arm_joint_ids(self) -> List[int]:
        return self._motion_planner.arm_joint_ids




    ################################################################################

    def __init__(self, config_file:str, objects:Collection[ObjectSpec], *, headless:bool=True, verbose=True):
        self._igibson_setup(config_file=config_file, headless=headless)
        self.load_objects(objects, verbose=verbose)
        self._init_object_state(objects,verbose=verbose)
        self._init_robot_state()

    
    def _igibson_setup(self, config_file:str, headless:bool=True) -> Tuple[iGibsonEnv, MotionPlanningWrapper]:
        config = self._load_config(config_file=config_file)
        if not config['robot']['name'] == 'Fetch':
            raise NotImplementedError("Only implemented for a single Fetch robot.")
        env = iGibsonEnv(
            config_file=config,
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

    def load_objects(self, specifications:Collection[Union[str,dict]], verbose:bool=True) -> None:
        '''Load objects into simulator based on a list of categories.
        ''' 
        body_ids_by_object = {}
        for spec in specifications:
            spec = _as_urdf_spec(spec)
            body_ids = self.load_object(spec=spec, verbose=verbose)
            body_ids_by_object[spec["name"]] = body_ids 
        return body_ids_by_object
    
    def _init_object_state(self, obj_specs:Collection[ObjectSpec], *, verbose=True) -> None:
        for spec in obj_specs:
            assert isinstance(spec, dict)
            if not isinstance(spec,ObjectSpec): 
                spec = ObjectSpec(**spec)
            for state in self._states:
                if state in spec:    
                    self.set_state(spec["name"], state, spec[state])
            
    def _init_robot_state(self) -> None:
        '''Initialize robot state. 
        
        WARNING: this steps sim; no objects can be added to scene after this is called.
        '''
        self.env.reset()
        self.env.land(self.env.robots[0], [0, 0, 0], [0, 0, 0]) # note: this steps sim!
        for robot in self.env.robots:
            robot.tuck() # tuck arm 
    
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
            lambda self, body,val: ft.partial(self.set_orientation, force_quaternion=True)(body,val)
        ),
    }
    assert set(_states) == set(_state_access_map)

    def get_state(self, body:UniqueID, state:str) -> Any:
        assert state in self._states
        return MyiGibsonSemanticInterface._state_access_map[state][0](self,body)
    
    def set_state(self, body:UniqueID, state:str, value:Any) -> None:
        assert state in self._states
        MyiGibsonSemanticInterface._state_access_map[state][1](self, body, value)

    # ----------------------------------------------------------------------

    def get_pose(self, body:UniqueID) -> Pose:
        try:
            return self.get_position_orientation(body)
        except KeyError as e:
            if body in ("eef", "gripper", self.eef_id):
                return self.get_eef_pose()
            else:
                raise e

        

    
    def set_pose(self, 
                 body:UniqueID, 
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
        # ignored return vals in order are: 
        #   localInertialFramePosition: Position3D, localInertialFrameOrientation: Quaternion,
        #   worldLinkFramePosition: Position3D, worldLinkFrameOrientation: Quaternion
        # In addition, if pybullet.getLinkState is called with computeLinkVelocity=1, there will be two additional return values:
        #   worldLinkLinearVelocity:Tuple[float,float,float], worldLinkAngularVelocity:Tuple[float,float,float]
        pos, orn, _, _, _, _ = pb.getLinkState(self.robot_id, self.eef_id, computeForwardKinematics=True, physicsClientId=get_client())
        return (pos, orn)
    

    def get_arm_config(self) -> JointPos:
        return get_joint_positions(self.robot_id, self._arm_joint_ids)
    def set_arm_config(self, q:JointPos) -> None:
        set_joint_positions(self.robot_id, self._arm_joint_ids, q)

    # ----------------------------------------------------------------------
    
    def is_movable(self, body: UniqueID):
        obj = self.get_object(body)
        return not obj.fixed_base

    def is_placement(self, body:UniqueID, surface:UniqueID):
        body    = self.get_object(body)
        surface = self.get_object(body)
        return body.states[object_states.OnTop].get_value(surface)
    
    ################################################################################

    def test_cfree_pose(self, 
                        body:UniqueID, 
                        pose:Optional[Pose]=None, 
                        body2:Optional[UniqueID]=None,
                        pose2:Optional[Pose]=None, 
                        *,
                        robot:Optional[Union[BaseRobot,UniqueID]]=None
                        ) -> bool:
        assert((pose2 is None) or (body2 is not None)), "cannot specify arg 'pose2' without arg 'body2'"
    
        if robot is None:
            robot = self._robot
        elif isinstance(robot, int):
            robot = self._robots[robot]
        elif isinstance(robot, str):
            robot = self.get_object(robot)
        assert isinstance(robot, BaseRobot)

        with UndoableContext(robot):
            
            body_links = self.get_object(body).get_body_ids()
            if isinstance(body,str): 
                body = self._get_bodyids_from_name(body)[0]
            
            if body2 is not None:
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
        
    def sample_arm_config(self, attachments=[], collisions=True, max_attempts=100):
        with UndoableContext(self._robot):
            robot_id = self._motion_planner.robot_id
            joint_ids = self._motion_planner.arm_joint_ids
            sample_fn = get_sample_fn(robot_id, joint_ids)
            config = sample_fn()
            if collisions:
                for _ in range(max_attempts):
                    set_joint_positions(robot_id, joint_ids, config)
                    obstacles = self.get_collidable_body_ids(include_robot=True)
                    if not any(pairwise_collision(robot_id, obs) for obs in obstacles):
                        return config
                return None
            else:
                return config

    def sample_placement(self, 
                         body:UniqueID, 
                         surface:UniqueID, 
                         robot:Optional[Union[BaseRobot,UniqueID]]=None
                         ) -> Pose:
        if robot is None:
            robot = self._robot
        elif isinstance(robot, int):
            robot = self._robots[robot]
        elif isinstance(robot, str):
            robot = self.get_object(robot)
        assert isinstance(robot, BaseRobot)

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
                skip_falling=True,
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
    
    def get_collidable_body_ids(self, include_robot:Union[bool,UniqueID,BaseRobot]=False):
        if include_robot is False: # could be the robot's UniqueID
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
                       body:UniqueID, 
                       obstacles:Collection[UniqueID]=[], 
                       *, 
                       exclude:Collection[UniqueID]=[], 
                       critical_obstacles:Collection[UniqueID]=[], 
                       max_distance:float=HAND_MAX_DISTANCE
                ) -> List[UniqueID]:
        if not obstacles and not critical_obstacles:
            obstacles = [obj for obj in self.objects if not (obj==body or obj in exclude)]
        obstacles = set(obstacles) - set(exclude)

        close_objects = set(x[0] for x in pb.getOverlappingObjects(*get_aabb(body)))
        close_obstacles = (close_objects & set(obstacles)) | set(critical_obstacles)
        collision_ids = [obs for obs in close_obstacles if pairwise_collision(body, obs, max_distance=max_distance)]
        return collision_ids

    def get_all_collisions(self, body:Union[UniqueID, Collection[UniqueID]], obstacles:Collection[UniqueID], *, critical_obstacles=[], max_distance:float=HAND_MAX_DISTANCE):
        collisions = {
            bid : self._get_collision(body=bid, obstacles=obstacles, critical_obstacles=critical_obstacles,max_distance=max_distance) 
            for bid in nonstring_iterable(body)
        }
        return collisions

    
    def get_robot_arm_collision_params(self,
                                *,
                                obstacles:List[int]=[],
                                ignore_other_scene_obstacles:bool=False,
                                self_collisions:bool=True, 
                                disabled_collisions:Set[Tuple[int,int]]={},
                                allow_collision_links:List[int]=[],
    ) -> Tuple[BaseRobot, int,List[int],List[int],Set[Tuple[int,int]],List[Tuple[int,int]],List[int],List[float],List[float]]:
        '''Assuming one Fetch robot so we can use MotionPlanningWrapper functionality
        '''
        robot     = self._motion_planner.robot
        robot_id  = self._motion_planner.robot_id
        joint_ids = self._motion_planner.arm_joint_ids
        if not ignore_other_scene_obstacles:
            obstacles = list(set(obstacles) | set(self._motion_planner.mp_obstacles))

        disabled_collisions |= robot.disabled_collision_pairs
        if robot.model_name == "Fetch":
            disabled_collisions |= {
                (link_from_name(robot_id, "torso_lift_link"), link_from_name(robot_id, "torso_fixed_link")),
                (link_from_name(robot_id, "torso_lift_link"), link_from_name(robot_id, "shoulder_lift_link")),
                (link_from_name(robot_id, "torso_lift_link"), link_from_name(robot_id, "upperarm_roll_link")),
                (link_from_name(robot_id, "torso_lift_link"), link_from_name(robot_id, "forearm_roll_link")),
                (link_from_name(robot_id, "torso_lift_link"), link_from_name(robot_id, "elbow_flex_link")),
            }
            allow_collision_links = list(
                set(allow_collision_links) | 
                set(self.eef_id) | 
                set([finger.link_id for finger in self._robot.finger_links[self._robot.default_arm]])
            )

        
        # Pair of links within the robot that need to be checked for self-collisions
        self_collision_id_pairs = get_self_link_pairs(robot_id, joint_ids, disabled_collisions) if self_collisions else []

        # List of links that move on the robot and that should be checked for collisions with the obstacles
        moving_links = frozenset([item for item in get_moving_links(robot_id, joint_ids) if not item in allow_collision_links])
        moving_robot_ids = [(robot_id, moving_links)]
        
        # Joint limits
        lower_limits, upper_limits = get_custom_limits(robot_id, joint_ids)

        return robot, robot_id, joint_ids, obstacles, disabled_collisions, self_collision_id_pairs, moving_robot_ids, lower_limits, upper_limits
    
    
    
    
    def arm_ik(self, eef_position:Position, eef_orientation:Optional[Orientation]=None, use_nullspace:bool=True, **kwargs) -> Optional[JointPos]:
        '''Calculate Fetch arm joint configuration that satisfies the given end-effector position in the workspace (ignoring collisions). 
        Can additionally specify a target end-effector orientation and/or joint configuration nullspace.
        '''
        # with UndoableContext(self._robot):   
        *null_space, joint_damping = self._motion_planner.get_ik_parameters()
        threshold = self._motion_planner.arm_ik_threshold
        # ik_solver = pb.IK_DLS or pb.IK_SDLS
        ik_spec =  {
            # robot/eef body IDs
            'bodyUniqueId' : self.robot_id, 
            'endEffectorLinkIndex' : self.eef_id, 
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
    
    COLLISION_INPUT_TYPES = Enum('COLLISION_INPUT_TYPES', [
            'JOINT_CONFIG', 
            'BODYCONF', 
            'EEF_POSITION', 
            'EEF_POSE',
            'BODYPOSE'
        ])
    

    def _isinstance_kinematic(self, x:Any, T:Type):
        '''May give false positives'''
        tests = {
            BodyConf :    (lambda x: isinstance(x,BodyConf)),
            BodyPose :    (lambda x: isinstance(x,BodyPose)),
            JointPos :    (lambda x: hasattr(x,"__len__") and len(x)==len(self._arm_joint_ids) and is_numeric_vector(x) ),
            Pose :        (lambda x: hasattr(x,"__len__") and len(x)==2 and is_numeric_vector(x[0])),
            Position :    (lambda x: hasattr(x,"__len__") and len(x)==3 and is_numeric_vector(x)),
        }
        return tests[T](x)


    def _as_arm_config(self, x, disabled_types=[]):
        disabled = lambda T: T in disabled_types
        if not disabled(BodyConf) and self._isinstance_kinematic(x,BodyConf):
            assert (x.body == self.robot_id),    f"BodyConf {x} field 'body' does not match body ID of {self.robot}."
            assert (x.joints == self._arm_joint_ids), f"BodyConf {x} field 'joints' does not match arm joint IDs of {self.robot}."
            joint_positions = x.configuration
        elif not disabled(BodyPose) and self._isinstance_kinematic(x,BodyPose):
            assert (x.body==self.robot_id) or (x.body==self.eef_id) 
            joint_positions = self.arm_ik(*x.pose)
        elif not disabled(Pose) and self._isinstance_kinematic(x,Pose):
            joint_positions = self.arm_ik(*x)
        elif not disabled(JointPos) and self._isinstance_kinematic(x,JointPos):
            joint_positions = x
        elif not disabled(Position) and self._isinstance_kinematic(x,Position):
            joint_positions = self.arm_ik(eef_position=x)
        else:
            raise TypeError(
                f"Could not interpret input {x} of type {type(x)} as a joint configuration or end effector pose. '"
                f"Expected one of: JointPos, BodyConf, or EEF BodyPose, Position, or Pose. "
            )
        assert self._isinstance_kinematic(joint_positions,JointPos)
        return joint_positions


    def _get_robot_arm_collision_fn(self,
                                *,
                                self_collisions:bool=True, 
                                disabled_collisions={},
                                allow_collision_links=[], 
                                **kwargs
    ):
        '''Assuming one Fetch robot so we can use MotionPlanningWrapper functionality
        '''
        ( 
            robot,
            robot_id, 
            joint_ids, 
            obstacles, 
            disabled_collisions, 
            self_collision_id_pairs, 
            moving_robot_ids, 
            lower_limits, 
            upper_limits 
        ) \
        = self.get_robot_arm_collision_params(
            self_collisions=self_collisions, 
            disabled_collisions=disabled_collisions,
            allow_collision_links=allow_collision_links, 
        )

        def _ensure_joint_position_vector(
            qspace_collision_fn:Callable[[JointPos,Iterable[Attachment]],bool]
        ) -> Callable[[KinematicConstraint,Iterable[Attachment]],bool]:
            ft.wraps(qspace_collision_fn)
            def _wrapper(x:KinematicConstraint, attachments:Iterable[Attachment]=[]) -> bool :
                joint_positions = self._as_arm_config(x)
                return qspace_collision_fn(q=joint_positions, attachments=attachments)
            return _wrapper
                    
        @_ensure_joint_position_vector
        def arm_collision_fn(q:JointPos, attachments:Iterable[Attachment]=[]) -> bool:
            assert len(joint_ids) == len(q)
            with UndoableContext(robot):
                if not all_between(lower_limits, q, upper_limits):
                    return True
                
                set_joint_positions(robot_id, joint_ids, q)
                for attachment in attachments:
                    attachment.assign()

                # Check for self collisions
                for link1, link2 in iter(self_collision_id_pairs):
                    if pairwise_link_collision(robot_id, link1, robot_id, link2):
                        return True
                
                # Include attachments as "moving bodies" to be tested for collisions
                attached_bodies = [attachment.child for attachment in attachments]
                moving_body_ids = moving_robot_ids + attached_bodies

                # Check for collisions of the moving bodies and the obstacles
                for moving, obs in product(moving_body_ids, obstacles):
                    if pairwise_collision(moving, obs, **kwargs):
                        return True
                
                # No collisions detected
                return False

        return arm_collision_fn


    def _get_hand_collision_fn(self, max_distance:float=HAND_MAX_DISTANCE):
        robot:BaseRobot = self._robot
        eef:UniqueID = self._eef.body_id
        non_hand_obstacles:Set[UniqueID] = set(obs for obs in self.get_collidable_bodies(include_robot=True) if (obs != eef))
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
    


    
    
    def plan_arm_joint_motion_pddlstream_style(self, 
        q1:JointPos, 
        q2:JointPos, 
        obstacles:List[int],
        attachments:List[Attachment]=[],
        weights:Optional[Iterable[float]]=None, 
        resolutions:Optional[Iterable[float]]=None, 
        algorithm:str='birrt',
        ignore_other_scene_obstacles=False,
        *, 
        planning_utils_version:PlanningUtilsVersion=PlanningUtilsVersion.PDDLSTREAM, 
        use_aabb:bool=False, 
        cache:bool=True, 
         **kwargs
    ):
        '''My attempt to reconcile plan_joint_motion between the implementations used by pddlstream and iGibson.
        '''
        
        assert len(self._arm_joint_ids) == len(q1) and len(self._arm_joint_ids) == len(q2)

        pbtools_module = PYBULLET_TOOLS_MODULES[planning_utils_version]
        distance_fn, sample_fn, extend_fn, collision_fn = self._get_arm_joint_motion_helper_fns(
            pbtools_module=pbtools_module, 
            obstacles=obstacles,
            ignore_other_scene_obstacles=ignore_other_scene_obstacles,
            attachments=attachments,
            weights=weights,
            resolutions=resolutions,
            use_aabb=use_aabb, 
            cache=cache,
        )
        
        with UndoableContext(self._robot):
            if not check_initial_end(q1, q2, collision_fn):
                return None
            
            motion_module = MOTION_PLANNING_MODULES[planning_utils_version]
            mp_algo_fn = self._get_motion_planning_algorithm(motion_module, algorithm)
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
    
    def _get_motion_planning_algorithm(self, motion_module, algorithm):
        if algorithm == 'direct':
            return motion_module.direct_path
        elif algorithm == 'birrt':
            return motion_module.birrt 
        elif algorithm == 'rrt_star':
            return motion_module.rrt_star 
        elif algorithm == 'rrt':
            return motion_module.rrt 
        elif algorithm == 'lazy_prm':
            return motion_module.lazy_prm_replan_loop 
        else:
            raise ValueError(f"Inappropriate argument algorithm={algorithm}. Expected one of: 'direct', 'birrt', 'rrt_star', 'rrt', or 'lazy_prm'")


    def _get_arm_joint_motion_helper_fns(self, 
        pbtools_module,
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
    ):
        ( 
            _,
            robot_id, 
            joint_ids, 
            obstacles, 
            disabled_collisions, 
            self_collisions, 
            _, 
            _, 
            _ 
        ) \
        = self.get_robot_arm_collision_params(
            self_collisions=self_collisions, 
            disabled_collisions=disabled_collisions,
            ignore_other_scene_obstacles=ignore_other_scene_obstacles
        )

        if pbtools_module==PYBULLET_TOOLS_MODULES[PlanningUtilsVersion.PDDLSTREAM]:
            if (weights is None) and (resolutions is not None):
                weights = np.reciprocal(resolutions)
            distance_fn = pbtools_module.get_distance_fn(robot_id, joint_ids, weights=weights)
            sample_fn = pbtools_module.get_sample_fn(robot_id, joint_ids)
            extend_fn = pbtools_module.get_extend_fn(robot_id, joint_ids, resolutions=resolutions)
            collision_fn = pbtools_module.get_collision_fn(
                body=robot_id, joints=joint_ids, obstacles=obstacles, attachments=attachments, 
                self_collisions=self_collisions, disabled_collisions=disabled_collisions,
                use_aabb=use_aabb, cache=cache
            )
        elif pbtools_module==PYBULLET_TOOLS_MODULES[PlanningUtilsVersion.IGIBSON]:
            distance_fn = pbtools_module.get_distance_fn(robot_id, joint_ids, weights=weights)
            sample_fn = pbtools_module.get_sample_fn(robot_id, joint_ids)
            extend_fn = pbtools_module.get_extend_fn(robot_id, joint_ids, resolutions=resolutions)
            collision_fn = pbtools_module.get_collision_fn(
                body=robot_id, joints=joint_ids, obstacles=obstacles, attachments=attachments, 
                self_collisions=self_collisions, disabled_collisions=disabled_collisions,
                allow_collision_links=allow_collision_links
            )
        else:
            raise ValueError(
                f"Inappropriate argument pbtools_module={pbtools_module} of type {type(pbtools_module)}. Expected either " 
                f"{PYBULLET_TOOLS_MODULES[PlanningUtilsVersion.PDDLSTREAM]} or {PYBULLET_TOOLS_MODULES[PlanningUtilsVersion.IGIBSON]}"
            )
        return distance_fn, sample_fn, extend_fn, collision_fn


    
    

    ################################################################################
    
    def get_stable_gen(self): # -> Generator[Pose, None, None]:
        def gen(body:UniqueID, surface:UniqueID):
            while True:
                try:
                    placement = self.sample_placement(body, surface)
                except ActionPrimitiveError:
                    continue
                if placement is not None:
                    yield BodyPose(body, placement)
        return gen
    
    

    def get_grasp_gen(self):        
        def grasps(body:UniqueID):
            target_obj:BaseObject = self.get_object(body)
            grasp_poses:List[Tuple[BodyPose,Orientation]] = get_grasp_poses_for_object(robot=self._robot, target_obj=target_obj, force_allow_any_extent=False)
            for grasp_pose, target_orn in grasp_poses:
                approach_position = grasp_pose[0] + target_orn * GRASP_APPROACH_DISTANCE
                approach_pose = (approach_position, grasp_pose[1])
                body_grasp = BodyGrasp(
                    body=self.get_id(target_obj), 
                    grasp_pose=grasp_pose,
                    approach_pose=approach_pose,
                    robot=self.robot_id,
                    link=self.eef_id
                )
                yield (body_grasp,)
        return grasps


    # def get_ik_fn(self, fixed:Optional[List[UniqueID]]=None):
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
        joint_ids = self._motion_planner.arm_joint_ids
        max_limits, min_limits, rest_position, joint_range, joint_damping = self._motion_planner.get_ik_parameters()
        
        def calculate_grasp_command(target:UniqueID, grasp:BodyGrasp) -> Tuple[JointPos,Command]:
            '''Calculate the actions needed to grasp the target object.
            
            Returns a tuple of a nearby 'approach' position for initiating the grasp, and
            a Command object containing a BodyPath from the approach to the grasp configuration, 
            an Attach action, and then the reversed BodyPath back to the approach pose; 
            or None if no valid path could be found in the number of attempts specified.

            :param target: UniqueID, name or body_id of target object to grasp
            :param grasp: BodyGrasp, grasp info for environment robot grasping target object with poses in world frame. 
            '''
            with UndoableContext(self._motion_planner.robot):
                target_id = self.get_id(target)

                # Some sanity checks
                assert isinstance(grasp, BodyGrasp), f"grasp must be defined as a BodyGrasp object, not {type(grasp)}"
                assert grasp.robot == self.robot_id, f"grasp must be defined for robot body ID {self.robot_id}, not {grasp.robot}" 
                assert grasp.link == self.eef_id, f"grasp must be defined for eef body ID {self.eef_id}, not {grasp.link}" 
                assert grasp.body == target_id, f"grasp must be defined for object body ID {target_id}, not {grasp.body}"

                ### Get joint angles for desired approach and grasp poses
                approach_position, _ = grasp.approach_pose
                approach_config = self.arm_ik(eef_position=approach_position)
                grasp_position, grasp_orientation = grasp.grasp_pose
                grasp_config = self.arm_ik(eef_position=grasp_position, eef_orientation=grasp_orientation)
                
                # Find a collision-free path from approach to grasp config
                for _ in range(num_attempts):
                    self.set_arm_config(approach_config)
                    path = self._motion_planner.plan_arm_motion(arm_joint_positions=grasp_config)
                    if path is not None:
                        # Starting at approach config, move to grasp config, perform grasp, and return to approach config
                        command = Command([ BodyPath(self.robot_id, path),
                                            Attach(target_id, self.robot_id, self.eef_id),
                                            BodyPath(self.robot_id, path[::-1], attachments=[grasp])])
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



    
    def assign_poses(self, fluents:Iterable[Tuple[UniqueID,...]]) -> Set[int]:
        obstacles = set()
        for fluent in fluents:
            name, *args = fluent
            if name.lower() == 'atpose':
                obj, pose = args
                obj = self.get_id(obj)
                assert isinstance(obj, int)
                if self._isinstance_kinematic(pose,BodyPose):
                    assert pose.body == obj
                    pose = pose.pose
                assert self._isinstance_kinematic(pose,Pose)
                obstacles.add(obj)
                self.set_pose(obj, pose)
            else:
                raise ValueError(name)
        return obstacles


    def get_free_motion_gen(self):
        robot = self._robot
        robot_id = self.robot_id
        joint_ids = self._arm_joint_ids
        def qfree_traj(q1:JointPos, q2:JointPos, atpose_fluents:List[Tuple]=[]) -> Optional[Command]:
            with UndoableContext(robot):
                # q1.assign()
                self.set_arm_config(q1)
                # Temporarily use only specified obstacles for motion planner
                original_mp_obstacles = self._motion_planner.mp_obstacles
                self._motion_planner.mp_obstacles = set(self.get_collidable_body_ids()) | self.assign_poses(atpose_fluents)
                # Calculate arm trajectory
                path = self._motion_planner.plan_arm_motion(q2)
                # Reassign old mp_obstacles to original value
                self._motion_planner.mp_obstacles = original_mp_obstacles
                # Return value
                if path is not None:
                    command = Command([BodyPath(robot_id, path, joints=joint_ids)]) if path is not None else None
                    return (command,)
                else:
                    return tuple()

        def _consistency_check(q:KinematicConstraint) -> None:
            if isinstance(q,BodyConf):
                assert (q.body == robot_id) and (q.joints == joint_ids)
            elif isinstance(q, BodyPose):
                assert (q.body == robot_id)

        ft.wraps(qfree_traj)
        def wrapper(q1:KinematicConstraint, q2:KinematicConstraint, atpose_fluents:List[Tuple]=[]) -> Optional[Command]:
            _consistency_check(q1)
            _consistency_check(q2)
            q1 = self._as_arm_config(q1)
            q2 = self._as_arm_config(q2)
            return qfree_traj(q1, q2, atpose_fluents=atpose_fluents)
        return wrapper


    def get_motion_gen(self):
        robot = self._motion_planner.robot
        robot_id = self._motion_planner.robot_id
        joint_ids = self._arm_joint_ids
        def qfree_traj(q1:JointPos, q2:JointPos, grasp:BodyGrasp=None, atpose_fluents:List[Tuple]=[]) -> Optional[Command]:
            with UndoableContext(robot):
                # q1.assign()
                self.set_arm_config(q1)
                self._get_robot_arm_collision_fn()
                # Temporarily use only specified obstacles for motion planner
                original_mp_obstacles = self._motion_planner.mp_obstacles
                self._motion_planner.mp_obstacles = set(self.get_collidable_body_ids()) | self.assign_poses(atpose_fluents)
                # Calculate arm trajectory
                path = self._motion_planner.plan_arm_motion(q2)
                # Reassign old mp_obstacles to original value
                self._motion_planner.mp_obstacles = original_mp_obstacles
                # Return value
                if path is not None:
                    command = Command([BodyPath(robot_id, path, joints=joint_ids)]) if path is not None else None
                    return (command,)
                else:
                    return tuple()

        def _consistency_check(q:KinematicConstraint) -> None:
            if isinstance(q,BodyConf):
                assert (q.body == robot_id) and (q.joints == joint_ids)
            elif isinstance(q, BodyPose):
                assert (q.body == robot_id)

        ft.wraps(qfree_traj)
        def wrapper(q1:KinematicConstraint, q2:KinematicConstraint, atpose_fluents:List[Tuple]=[]) -> Optional[Command]:
            _consistency_check(q1)
            _consistency_check(q2)
            q1 = self._as_arm_config(q1)
            q2 = self._as_arm_config(q2)
            return qfree_traj(q1, q2, atpose_fluents=atpose_fluents)
        return wrapper 



    # def get_holding_motion_gen(self, robot, fixed=[], teleport=False, self_collisions=True):
    #     pass

    # def get_movable_collision_test(self):
    #     pass

    # def get_joint_states(self, body: UniqueID):
        # only using joint position, not velocity
        # return [joint[0] for joint in super().get_joint_states(body)]

    def get_cfree_pose_pose_test(self):
        return self.test_cfree_pose
    
    def get_cfree_approach_obj_pose_test(self):
        def collision_test(b1:UniqueID, p1:Pose, g1:BodyGrasp, b2:UniqueID, p2:Pose) -> bool:
            '''Determine if, when performing grasp g1 on target object b1 with pose p1,
            retracting back from grasp.grasp_pose to grasp.approach_pose will cause b1
            to collide with obstacle object b2 with pose p2.
            '''
            if (b1 == b2): 
                return False
            with UndoableContext(self._robot):
                self.set_pose(b2, p2)
                for obj_pose in interpolate_poses(g1.grasp_pose, g1.approach_pose):
                    collision_free = self.test_cfree_pose(b1,obj_pose,b2,p2)
                    if not collision_free:
                        return True
                    # self.set_pose(b1, obj_pose)
                    # if pairwise_collision(b1, b2):
                    #     return True
                return False
        
        def _preprocess_inputs(b1:UniqueID, p1:Union[Pose,BodyPose], g1:BodyGrasp, 
                               b2:UniqueID, p2:Union[Pose,BodyPose]
                            ) -> Tuple[int,Pose,BodyGrasp,int,Pose]:
            if not isinstance(b1, int): 
                b1 = self.get_id(b1)
            if not isinstance(b2, int): 
                b2 = self.get_id(b2)
            if isinstance(p1, BodyPose):
                assert p1.body == b1
                p1 = p1.pose
            if isinstance(p2, BodyPose):
                assert p2.body == b2
                p2 = p2.pose
            assert g1.body == b1
            return b1,p1,g1,b2,p2

        def collision_free_grasp_retrieval_test(b1:UniqueID, p1:Union[Pose,BodyPose], g1:BodyGrasp, 
                                                b2:UniqueID, p2:Union[Pose,BodyPose]) -> bool:
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
        
        def _preprocess_inputs(command:Command, body:UniqueID, pose:Union[Pose,BodyPose]) -> Tuple[Command,int,Pose]:
            if not isinstance(body,int): body = self.get_id(body)
            if isinstance(pose,BodyPose):
                assert pose.body == body
                pose = pose.pose
            return command, body, pose
        
        def test(command:Command, body:UniqueID, pose:Union[Pose,BodyPose]) -> bool:
            '''Returns true if the moving bodies of 'command' DO NOT COLLIDE with 'body' located at pose 'pose'.
            '''
            command, body, pose = _preprocess_inputs(command, body, pose)
            collision = collision_test(command, body, pose)
            return collision
        
        return test

    ################################################################################

    


# TODO: put this in objspec file
def _as_urdf_spec(spec:Union[str,dict]) -> "URDFObjectSpec":
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


    
    
