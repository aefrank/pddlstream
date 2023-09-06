from collections.abc import Set
from collections import UserDict
import functools as ft
from itertools import product
import os
import sys
from typing import Any, Callable, Collection, Generator, Iterator, List, Optional, Tuple, Union

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
    ObjectSpec,
    Orientation3D,
    Position3D, 
    URDFObjectSpec,
    Orientation, 
    Position,
    UniqueID, 
)
from examples.pybullet.utils.pybullet_tools.kuka_primitives import (
    Attach,
    BodyGrasp,
    BodyPath, 
    BodyPose,
    Command,
    get_ik_fn, 
)
from examples.pybullet.utils.pybullet_tools.utils import Attachment, approach_from_grasp, end_effector_from_body, get_joint_positions, get_sample_fn, inverse_kinematics_helper, plan_direct_joint_motion, plan_joint_motion, plan_waypoints_joint_motion


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


    
def is_nonstring_iterable(x:Any) -> bool:
    return hasattr(x, "__iter__") and not isinstance(x, str)
    
def nonstring_iterable(x:Any) -> Iterator:
    if is_nonstring_iterable(x): return iter(x)
    elif isinstance(x,str): return iter([x])
    else: return iter(x)



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
    def _get_object(self, obj:Union[UniqueID,BaseObject]) -> BaseObject:
        assert isinstance(obj,(int,str,BaseObject,BaseRobot)), f"Invalid argument 'obj' of type {type(obj)}: {obj}"
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
        return self.env.scene.objects_by_id[body_id].name
        

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


    

class MyiGibsonSemanticInterface(iGibsonSemanticInterface):

    @property
    def _robot(self) -> BaseRobot:
        assert len(self._robots)==1, \
            f"attribute '_robot' undefined for multi-robot {self.__class__.__name__} containing {len(self._robots)} robots: {self._robots}. "
        return self._robots[0]
    @property
    def robot(self) -> UniqueID:
        assert len(self.robots)==1, \
            f"attribute 'robot' undefined for multi-robot {self.__class__.__name__} containing robots: {self.robots}. "
        return self.robots[0]

    @property
    def _eef(self) -> RobotLink:
        return self._robot._links[self.eef]
    @property
    def eef(self) -> str:
        return self._robot.eef_link_names[self._robot.default_arm]

    ################################################################################

    def __init__(self, config_file:str, objects:Collection[ObjectSpec], *, headless:bool=True, verbose=True):
        self._igibson_setup(config_file=config_file, headless=headless)
        self.load_objects(objects, verbose=verbose)
        self._init_object_state(objects,verbose=verbose)
        self._init_robot_state()

    
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

    def get_pose(self, body:UniqueID) -> Tuple[Position, Orientation]:
        return self.get_position_orientation(body)
    
    def set_pose(self, 
                 body:UniqueID, 
                 pose:Optional[Tuple[Position,Orientation]]=None, 
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


    # ----------------------------------------------------------------------
    
    def is_movable(self, body: UniqueID):
        obj = self._get_object(body)
        return not obj.fixed_base

    def is_placement(self, body:UniqueID, surface:UniqueID):
        body    = self._get_object(body)
        surface = self._get_object(body)
        return body.states[object_states.OnTop].get_value(surface)
    
    ################################################################################

    def test_cfree_pose(self, 
                        body:UniqueID, 
                        pose:Optional[Tuple[Position,Orientation]]=None, 
                        body2:Optional[UniqueID]=None,
                        pose2:Optional[Tuple[Position,Orientation]]=None, 
                        *,
                        robot:Optional[Union[BaseRobot,UniqueID]]=None
                        ) -> bool:
        assert((pose2 is None) or (body2 is not None)), "cannot specify arg 'pose2' without arg 'body2'"
    
        if robot is None:
            robot = self._robot
        elif isinstance(robot, int):
            robot = self._robots[robot]
        elif isinstance(robot, str):
            robot = self._get_object(robot)
        assert isinstance(robot, BaseRobot)

        with UndoableContext(robot):
            
            body_links = self._get_object(body).get_body_ids()
            if isinstance(body,str): 
                body = self._get_bodyids_from_name(body)[0]
            
            if body2 is not None:
                body2_links = self._get_object(body2).get_body_ids()
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
        
        
    
    def sample_placement(self, 
                         body:UniqueID, 
                         surface:UniqueID, 
                         robot:Optional[Union[BaseRobot,UniqueID]]=None
                         ) -> Tuple[Position, Orientation]:
        if robot is None:
            robot = self._robot
        elif isinstance(robot, int):
            robot = self._robots[robot]
        elif isinstance(robot, str):
            robot = self._get_object(robot)
        assert isinstance(robot, BaseRobot)

        obj = self._get_object(body)

        with UndoableContext(robot):
            # pred_map = {object_states.OnTop: "onTop", object_states.Inside: "inside"}
            result = sample_kinematics(
                predicate="onTop",
                objA=obj,
                objB=self._get_object(surface),
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
                else self._get_object(robot)
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

    
    def _get_robot_arm_collision_fn(self,
                                *,
                                self_collisions:bool=True, 
                                disabled_collisions={},
                                allow_collision_links=[], 
                                **kwargs
    ):
        '''Assuming one Fetch robot so we can use MotionPlanningWrapper functionality
        '''
        robot = self.robot
        robot_id = self._motion_planner.robot_id
        arm = self._motion_planner.arm_joint_ids
        obstacles = self._motion_planner.mp_obstacles

        disabled_collisions |= robot.disabled_collision_pairs
        if robot.model_name == "Fetch":
            disabled_collisions |= {
                (link_from_name(robot_id, "torso_lift_link"), link_from_name(robot_id, "torso_fixed_link")),
                (link_from_name(robot_id, "torso_lift_link"), link_from_name(robot_id, "shoulder_lift_link")),
                (link_from_name(robot_id, "torso_lift_link"), link_from_name(robot_id, "upperarm_roll_link")),
                (link_from_name(robot_id, "torso_lift_link"), link_from_name(robot_id, "forearm_roll_link")),
                (link_from_name(robot_id, "torso_lift_link"), link_from_name(robot_id, "elbow_flex_link")),
            }

        
        # Pair of links within the robot that need to be checked for self-collisions
        self_collision_pairs = get_self_link_pairs(robot, arm, disabled_collisions) if self_collisions else []

        # List of links that move on the robot and that should be checked for collisions with the obstacles
        moving_links = frozenset([item for item in get_moving_links(robot, arm) if not item in allow_collision_links])
        moving_bodies = [(robot, moving_links)]
        
        # Joint limits
        lower_limits, upper_limits = get_custom_limits(robot, arm)

        def arm_collision_fn(pose3d:Tuple[Position3D,Orientation3D], attachments:List[Attachment]=[]):
            with UndoableContext(self._motion_planner.robot):
            
                q = self._motion_planner.get_arm_joint_positions(arm_ik_goal=pose3d)
                if not all_between(lower_limits, q, upper_limits):
                    return True
                
                set_joint_positions(robot, arm, q)
                for attachment in attachments:
                    attachment.assign()

                # Check for self collisions
                for link1, link2 in iter(self_collision_pairs):
                    if pairwise_link_collision(robot, link1, robot, link2):
                        return True
                
                # Include attachments as "moving bodies" to be tested for collisions
                attached_bodies = [attachment.child for attachment in attachments]
                moving_bodies += attached_bodies

                # Check for collisions of the moving bodies and the obstacles
                for moving, obs in product(moving_bodies, obstacles):
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
    
    ################################################################################
    
    def get_stable_gen(self): # -> Generator[Tuple[Position, Orientation], None, None]:
        def gen(body:UniqueID, surface:UniqueID):
            while True:
                placement = self.sample_placement(body, surface)
                if placement is not None:
                    yield BodyPose(body, placement)
        return gen
    
      
    
    def get_grasp_gen(self, robot:Optional[UniqueID]=None):
        robot:BaseRobot = self._get_object(robot) if robot is not None else self._robot
        def gen(body:UniqueID):
            target_obj:BaseObject = self._get_object(body)
            grasp_poses:List[Tuple[BodyPose,Orientation]] = get_grasp_poses_for_object(robot=robot, target_obj=target_obj, force_allow_any_extent=False)
            for grasp_pose, target_orn in grasp_poses:
                approach_position = grasp_pose[0] + target_orn * GRASP_APPROACH_DISTANCE
                approach_pose = (approach_position, grasp_pose[1])
                body_grasp = BodyGrasp(
                    body=self._get_bodyids_from_obj(target_obj)[0], 
                    grasp_pose=grasp_pose,
                    approach_pose=approach_pose,
                    robot=self._get_bodyids_from_obj(robot)[0],
                    link=robot.eef_links[robot.default_arm].link_id
                )
                yield (body_grasp,)
        return gen


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
    #         print('POSE    \t\t',pose.body, pose.pose)
    #         x,y,z = grasp.grasp_pose[0]
    #         a,b,c,d = grasp.grasp_pose[1]
    #         u,v,w = grasp.approach_pose[0]
    #         p,q,r,s = grasp.approach_pose[1]
    #         grasp_body = self._get_name_from_bodyid(grasp.body)
    #         grasp_robot = self._get_name_from_bodyid(grasp.robot)
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


    def get_grasp_plan_fn(self, num_attempts=10):
        robot_id = self._motion_planner.robot_id
        joint_ids = self._motion_planner.arm_joint_ids
        max_limits, min_limits, rest_position, joint_range, joint_damping = self._motion_planner.get_ik_parameters()
        
        def fn(target:str, grasp:BodyGrasp) -> Optional[Command]:
            '''Calculate the actions needed to grasp the target object.
            
            Returns a tuple of a nearby 'approach' position for initiating the grasp, and
            a Command object containing a BodyPath from the approach to the grasp configuration, 
            an Attach action, and then the reversed BodyPath back to the approach pose; 
            or None if no valid path could be found in the number of attempts specified.
            '''
            with UndoableContext(self._robot):
                eef_id = grasp.link
                target_id = self._get_bodyids_from_name(target)[0]

                ### Get joint angles for desired grasp pose
                grasp_pose = end_effector_from_body(self.get_pose(target), grasp.grasp_pose)
                grasp_position, grasp_orientation = grasp_pose
                # IK
                grasp_config = pb.calculateInverseKinematics(
                    robot_id,
                    eef_id,
                    targetPosition=grasp_position,
                    targetOrientation=grasp_orientation,
                    lowerLimits=min_limits,
                    upperLimits=max_limits,
                    jointRanges=joint_range,
                    restPoses=rest_position,
                    jointDamping=joint_damping,
                    # solver=p.IK_DLS,
                    maxNumIterations=100,
                )
                # Extract joint angles from full configuration
                grasp_config = tuple(np.array(grasp_config)[self._motion_planner.robot_arm_indices])
                
                ### Get joint angles for start configuration from grasp approach pose 
                approach_pose = approach_from_grasp(grasp.approach_pose, grasp_pose)
                approach_position, _ = approach_pose # we don't need to constrain orientation for approach pose
                approach_config = tuple(self._motion_planner.get_arm_joint_positions(approach_position))

                # Confirm a collision-free path exists from approach to grasp config
                for _ in range(num_attempts):
                    set_joint_positions(robot_id, joint_ids, approach_config)
                    path = self._motion_planner.plan_arm_motion(arm_joint_positions=grasp_config)
                    
                    if path is not None:
                        command = Command([ BodyPath(robot_id, path),
                                            Attach(target_id, robot_id, grasp.link),
                                            BodyPath(robot_id, path[::-1], attachments=[grasp])])
                        return (approach_config, command)
        return fn

            


    def get_free_motion_gen(self, robot, fixed=[], teleport=False, self_collisions=True):
        pass

    def get_holding_motion_gen(self, robot, fixed=[], teleport=False, self_collisions=True):
        pass

    def get_movable_collision_test(self):
        pass

    # def get_joint_states(self, body: UniqueID):
        # only using joint position, not velocity
        # return [joint[0] for joint in super().get_joint_states(body)]

    def get_cfree_pose_pose_test(self):
        return self.test_cfree_pose

    ################################################################################

    



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


    
    
