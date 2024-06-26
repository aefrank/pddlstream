import sys, os
import subprocess
from argparse import ArgumentError, ArgumentParser, Namespace
from collections import UserDict
from itertools import product, takewhile
from json import load
import os
import functools as ft
import inspect as ins
from time import sleep
import cv2
from typing_extensions import TypeAlias
from igibson import object_states
from igibson.envs.igibson_env import iGibsonEnv
from igibson.robots import Fetch
from igibson.utils.grasp_planning_utils import get_hand_rotation_from_axes
import numpy as np
from typing import Any, Callable, List, NamedTuple, NamedTuple, NewType, NoReturn, Optional, Tuple, Iterable, Union

from igibson.utils.assets_utils import get_all_object_categories, get_ig_model_path, get_object_models_of_category
from numpy import argsort, pi, random
from transforms3d.euler import quat2euler
from examples.fetch.from_kuka_tamp.utils.utils import PybulletToolsVersion, import_module
from iGibson.igibson.simulator import load_without_pybullet_vis
from iGibson.igibson.utils.assets_utils import get_ig_avg_category_specs

from igibson.action_primitives.starter_semantic_action_primitives import URDFObject, UndoableContext
from igibson.objects.object_base import BaseObject

from examples.fetch.from_kuka_tamp.fetch_primitives import BID, UID, JointPos, KinematicConstraint, MyiGibsonSemanticInterface, Object, iGibsonSemanticInterface #, _sync_viewer_after_exec
from examples.fetch.from_kuka_tamp.utils.object_spec import Euler, Kwargs, ObjectSpec, Orientation3D, Position3D, Quaternion
from examples.pybullet.utils.pybullet_tools.kuka_primitives import ApplyForce, Attach, BodyConf, BodyGrasp, BodyPose, BodyPath, Command
from pddlstream.algorithms.meta import create_parser, solve
from pddlstream.language.constants import PDDLProblem
from pddlstream.language.generator import from_fn, from_gen_fn, from_list_fn, from_test
from pddlstream.utils import Profiler, read

from igibson.external.pybullet_tools.utils import refine_path, wait_for_user
from examples.pybullet.utils.pybullet_tools.utils import Attachment, CollisionInfo, CollisionPair, wait_for_duration


##############################################################################

Pose: TypeAlias = Tuple[Position3D, Orientation3D]


def prune_kwargs(fn):
    fn_kwargs = [k for k,param in ins.signature(fn).parameters.items() if param.kind in [ins.Parameter.KEYWORD_ONLY, ins.Parameter.POSITIONAL_OR_KEYWORD, ins.Parameter.VAR_KEYWORD]]
    # fn_varkwargs = [k for k,param in ins.signature(fn).parameters.items() if param.kind==ins.Parameter.VAR_KEYWORD]
    @ft.wraps(fn)
    def wrapper(*args, **kwargs):
        relevant_kwargs = {k:v for k,v in kwargs.items() if k in fn_kwargs}
        return fn(*args, **relevant_kwargs)
    return wrapper

def _convert_args_to_raw_arm_configs(*joint_pos_arg_ids, sim:Optional['ModifiedMiGSI']=None):
    for qarg in joint_pos_arg_ids:
        assert isinstance(qarg,(int,str)), f"Expected integer or string arguments, not {qarg} : {type(qarg)}."
    as_arm_config = lambda sim,x: sim.as_arm_config(x)

    def _decorator(
        qspace_fn:Callable[[JointPos,Iterable[Attachment]],bool]
    ) -> Callable[[KinematicConstraint,Iterable[Attachment]],bool]:
        ft.wraps(qspace_fn)
        sig = ins.signature(qspace_fn)
        def _wrapper(*args, **kwargs) -> bool :
            nonlocal sim
            if sim is None:
                sim = args[0]
            params = sig.bind(*args, **kwargs)
            params.apply_defaults()
            args, kwargs = params.args, params.kwargs
            for qarg in joint_pos_arg_ids:
                if isinstance(qarg,int):
                    args[qarg] = as_arm_config(sim, args[qarg])
                else:
                    params.arguments[qarg] = as_arm_config(sim, params.arguments[qarg])

            return qspace_fn(*params.args, **params.kwargs)
        
        return _wrapper
    return _decorator


##############################################################################


class ObjSpec(UserDict):
    URDF_ARGS = list(ft.reduce( lambda d1,d2: d1 | d2, 
        [set(ins.getfullargspec(c)[0]) 
         for c in ins.getmro(URDFObject) 
         if issubclass(c,BaseObject)]
    ))
    URDF_ARGS.remove('self')

    def __init__(self, name, category, **kwargs:Kwargs):
        self.data = {'name':str(name).strip(), 'category':str(category).strip()}
        
        kwargs.update(self.extract_file_info(category, **kwargs))
        if 'model' in kwargs: kwargs.pop('model')
        if 'scale' in kwargs and not isinstance(kwargs['scale'], np.ndarray):
            val = kwargs['scale']
            kwargs['scale'] = np.array(val) if not isinstance(val, (int,float)) else val*np.ones(3)

        self.data.update(kwargs)
        self._urdf_keys  = [key for key in self.data if key in ObjSpec.URDF_ARGS]
        self._state_keys = [key for key in self.data if key not in ObjSpec.URDF_ARGS]

    def extract_file_info(self, category, *, model=None, model_path=None, **kwargs):
        if category not in get_all_object_categories():
            raise ValueError(f"Unable to find object category '{category}' in assets.")
        if model is None: 
            model = random.choice(get_object_models_of_category(category))
        if model_path is None:
            model_path = get_ig_model_path(self.data["category"], model)
        filename = os.path.join(model_path, model + ".urdf")
        return {'filename' : filename, 'model_path': model_path}

    @property
    def urdf_kwargs(self):
        return {k:self.data[k] for k  in self._urdf_keys}
    
    @property
    def state(self):
        return {k:self.data[k] for k  in self._state_keys}

    
    _RAISED_ERROR_PASSER = NamedTuple('_RAISED_ERROR_PASSER', [('error', Exception)])
    def __get_attr_or_err(self, attr:str) -> Any: 
        try:
            return object.__getattribute__(self, attr)
        except AttributeError as e:
            return self.__class__._RAISED_ERROR_PASSER(e)
    
    def __getattr__(self, attr):
        '''
        Priority 1: if self.attr exists, return self.attr
        Priority 2: if attr is not 'data', self.data exists, and self.data[attr] exists, return self.data[attr]
        Priority 3: raise AttributeError
        '''
        value = self.__get_attr_or_err(attr)
        if not isinstance(value,self.__class__._RAISED_ERROR_PASSER):
            return value
        elif isinstance(value.error,AttributeError): 
            if  attr!='data'  \
            and not isinstance((data:=self.__get_attr_or_err('data')), self.__class__._RAISED_ERROR_PASSER) \
            and attr in data:
                return data[attr]
        # couldn't resolve error 
        raise value.error
        

    def __setattr__(self, attr, value):
        '''
        Priority 1: if self.attr exists, set self.attr=value
        Priority 2: if self.attr does not exist, attr is not 'data', and self.data[attr] exists, set self.data[attr]=value
        Priority 3: if neither exist, let a new attribute with name attr be created on self

        Note that Priority 1 and Priority 3 are achieved with the same line of code: object.__setattr__(self, attr, value)
        '''
        # check if self.attr exists
        current_value = self.__get_attr_or_err(attr)

        # If error raised while trying to access self.attr -> attempt to return via Priority 2
        # If no error raised while trying to access self.attr -> skip this diversion and continue to Priority 1
        if isinstance(current_value, self.__class__._RAISED_ERROR_PASSER):
            if not isinstance(current_value.error, AttributeError): 
                # if error raised not an AttributeError, something unexpected went wrong; raise error
                raise current_value.error            
            else:
                # error raised was an AttributeError -> move to Priority 2
                if attr!='data' and \
                    not isinstance((data:=self.__get_attr_or_err('data')), self.__class__._RAISED_ERROR_PASSER) and \
                    attr in data:
                        self.data[attr] = value
                        return
                # if we get here, attr not a member of data (or data doesn't exist) -> move to Priority 3

        # Either no error was raised when accessing attr directly (skipping the above 'if' block) and we can set its value directly, or 
        # attr was not a member of data dict (passing through 'if' block without reaching 'raise' or 'return' statements) and we must
        # create a new attribute on self; both cases are handled by:
        object.__setattr__(self, attr, value)
            

class ModifiedMiGSI(MyiGibsonSemanticInterface):

    @property
    def _objects(self):
        return [obj for obj in self.env.scene.get_objects() if not obj.category in ["agent","walls","floors","ceilings"]]
    @property 
    def objects(self):
        return [obj.name for obj in self._objects]

    def __init__(self, config_file:str, objects:Iterable[ObjSpec]=[], headless:bool=True, tensor:bool=False, let_settle:bool=False,
                 *, robot_pose:Pose=((0,0,0),(0,0,0)), viewer_pose:Optional[Pose]=None, verbose=True, **config_options:Kwargs):
        self.has_gui = not headless
        self._init_igibson(config=config_file, headless=headless, tensor=tensor, **config_options)
        self._init_objects(objects)
        self._init_robot_state(*robot_pose)
        if not headless:
            self._viewer_setup(*viewer_pose) if viewer_pose is not None else self._viewer_setup()
        if let_settle:
            print("Allowing to settle... ", end="")
            # for _ in range( int( 1 / self.env.action_timestep )):
            #     self.step()
            self.run_physics(3, sync_at_end=self.has_gui)
            self.step()
            print("done.")

        


    def run_physics(self, seconds, sync_at_end=True):
        import pybullet as pb

        for _ in range(int(seconds / self.env.simulator.physics_timestep)):
            pb.stepSimulation()

        self.env.simulator._non_physics_step()
        if sync_at_end:
            self.env.simulator.sync()
            self.env.simulator.frame_count += 1

    @load_without_pybullet_vis
    def _load_objects(self, specs:Iterable[ObjSpec], verbose: bool = True) -> List[BID]:
        def load_single_object(spec:ObjSpec):
            if verbose: print(f"Loading {spec['category'].capitalize()} object '{spec['name']}'... ", end='')  

            URDF_kwargs = {
                'avg_obj_dims' : get_ig_avg_category_specs().get(spec['category']),
                'fit_avg_dim_volume' : False,
                'texture_randomization' : False,
                'overwrite_inertial' : True,
            }
            URDF_kwargs.update(**spec.urdf_kwargs)
            
            # Create and import the object
            sim_obj = URDFObject(**URDF_kwargs)
            self.env.simulator.import_object(sim_obj)
            if verbose:  print(" done.")
            
            # Return obj body_id
            return self.get_bid(sim_obj)
        return [load_single_object(spec) for spec in specs]
    
    def load_object(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError
    def load_objects(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError

    def _init_object_statevars(self, 
                      name:str,
                      position:Position3D=(0,0,0), 
                      orientation:Orientation3D=Euler(0,0,0), 
                      place_on:Optional[str]=None, 
                      **state:Kwargs) -> None:
        if place_on is not None:
            # self.land(name, position, orientation)
            obj = self.get_object(name)
            surface = self.get_object(place_on)

            # position, orientation = self.sample_placement(name, place_on)
            # print(position)
            self.land(obj, position, orientation) #, _sync_viewer=False)
            # print(position)

            assert obj.states[object_states.OnTop].get_value(surface) #, use_ray_casting_method=True)
        else:
            self.set_position_orientation(name, position, orientation) #, _sync_viewer=False)
        if not state=={}:
            raise NotImplementedError(f"No handling defined for state variables {list(state)}")
    
    def _init_objects(self, specs:Iterable[ObjSpec]) -> None:
        self._load_objects(specs)
        
        pose = self.get_pose(self.robot)
        self.set_position(self.robot, (100,100,100)) #, _sync_viewer=False)
        for spec in specs:
            self._init_object_statevars(spec.name, **spec.state)
        self.set_pose(self.robot, pose) #, _sync_viewer=False)
    
    def _init_object_state(self, obj_specs: Iterable[ObjectSpec], *, verbose=True) -> None:
        raise NotImplementedError


    

    def step(self, *, action:np.ndarray=None):
        if action is not None:
            self.env.step(action)
        else:
            self.env.simulator.step()

    def is_surface(self, obj:UID):
        return not self.is_movable(obj) and (self.get_object(obj).category.lower() not in ['walls', 'ceilings'])
    
    # @_sync_viewer_after_exec
    def land(self, obj:Object, position:Optional[Position3D]=None, orientation:Optional[Orientation3D]=None):
        if position is None:    position    = self.get_position(obj)
        if orientation is None: orientation = self.get_orientation(obj)
        if len(orientation)==4: orientation = quat2euler(orientation)

        obj = self.get_object(obj)
        # print(f"{self.__class__.__name__}.land() called on object '{obj.name}'")
        if obj.fixed_base:
            print(f"WARNING: {self.__class__.__name__}.land() called on object '{obj.name}' with fixed_base=True.")
        self.env.land(obj, position, orientation)


    ###############################################################

    def _get_robot_arm_collision_fn(self,
                                *,
                                obstacles=None,
                                self_collisions=True, 
                                disabled_collisions={},
                                allow_collision_links=[], 
                                allow_gripper_collisions=False,
                                planning_utils_version=PybulletToolsVersion.PDDLSTREAM,
                                **kwargs
    ):
        if obstacles is None:
            obstacles = []
            ignore_other_scene_obstacles = False
        else:
            ignore_other_scene_obstacles = True
        pybullet_tools = import_module("pybullet_tools", planning_utils_version)

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
            obstacles=obstacles,
            ignore_other_scene_obstacles=ignore_other_scene_obstacles,
            self_collisions=self_collisions, 
            disabled_collisions=disabled_collisions,
            allow_collision_links=allow_collision_links, 
            allow_gripper_collisions=allow_gripper_collisions,
        )

    
    def get_robot_arm_collision_params(self, *args, **kwargs):
        raise NotImplementedError

    def _get_arm_collision_params(
        self, 
        obstacles=None,
        *,
        ignore_other_scene_obstacles=None,
        ignore_self_collisions=False,
        ignore_gripper_collisions=False,
        enforce_joint_limits=True,
        use_aabb=False,
        planning_utils_version=PybulletToolsVersion.PDDLSTREAM,
        **kwargs,
    ) -> bool:
        
        get_relevant_kwargs = lambda fn: {k:v for k,v in kwargs.items() if k in ins.signature(fn).parameters}

        if obstacles is None and ignore_other_scene_obstacles is None: 
            # neither specified -> default to using all obstacles in scene
            obstacles = []
            ignore_other_scene_obstacles = False
        elif ignore_other_scene_obstacles is None: 
            # obstacles specified, ignore_other_scene_obstacles is not; assume we ignore unspecified obstacles
            ignore_other_scene_obstacles = True
        elif obstacles is None: 
            # ignore_other_scene_obstacles specified, obstacles not; obstacles is empty list
            obstacles = []

        relevant_kwargs = get_relevant_kwargs(super().get_robot_arm_collision_params)
        ( 
            robot,
            robot_id, 
            joint_ids, 
            obstacles, 
            _, 
            _,
            self_collision_id_pairs, 
            moving_robot_bids, 
            lower_limits, 
            upper_limits 
        ) \
        = super().get_robot_arm_collision_params(
            obstacles=obstacles,
            ignore_other_scene_obstacles=ignore_other_scene_obstacles,
            self_collisions=not ignore_self_collisions, 
            allow_gripper_collisions=ignore_gripper_collisions,
            **relevant_kwargs
        )
        
        config_type_guard = _convert_args_to_raw_arm_configs
        config_setter = ft.partial(self.set_arm_config) #, _sync_viewer=False)
        pbtools_utils = import_module("pybullet_tools.utils", planning_utils_version)

        if enforce_joint_limits:
            relevant_kwargs = get_relevant_kwargs(pbtools_utils.all_between)
            violates_limits = lambda q: pbtools_utils.all_between(lower_limits, q, upper_limits, **relevant_kwargs)
        else:
            violates_limits = lambda q: False

        if use_aabb:
            # aabb functionality only in pddlstream pybullet_tools version
            aabb = pbtools_utils if planning_utils_version is PybulletToolsVersion.PDDLSTREAM else import_module("pybullet_tools.utils", PybulletToolsVersion.PDDLSTREAM)
            get_moving_aabb = lambda bid: aabb.cached_fn(aabb.get_buffered_aabb, cache=True, max_distance=aabb.MAX_DISTANCE/2., **kwargs)(bid)
            pairwise_link_collision = lambda l1, l2: \
                aabb.aabb_overlap(get_moving_aabb(robot_id), get_moving_aabb(robot_id)) \
                and pbtools_utils.pairwise_link_collision(robot_id, l1, robot_id, l2, **get_relevant_kwargs(pbtools_utils.pairwise_link_collision)) 
            pairwise_collision = lambda b1, b2: \
                aabb.aabb_overlap(get_moving_aabb(b1), get_moving_aabb(b2)) \
                and pbtools_utils.pairwise_collision(b1, b2, **get_relevant_kwargs(pbtools_utils.pairwise_collision))
        else:
            pairwise_link_collision = lambda l1,l2: \
                pbtools_utils.pairwise_link_collision(robot_id, l1, robot_id, l2, **get_relevant_kwargs(pbtools_utils.pairwise_link_collision)) 
            pairwise_collision = lambda b1, b2: \
                pbtools_utils.pairwise_collision(b1, b2, **get_relevant_kwargs(pbtools_utils.pairwise_collision))

        return robot, joint_ids, obstacles, self_collision_id_pairs, moving_robot_bids, config_type_guard, config_setter, violates_limits, pairwise_link_collision, pairwise_collision

        

    def _get_arm_collision_fn(
        self, 
        obstacles=None,
        *,
        ignore_other_scene_obstacles=None,
        ignore_self_collisions=False,
        ignore_gripper_collisions=False,
        attachments=[],
        enforce_joint_limits=True,
        use_aabb=False,
        planning_utils_version=PybulletToolsVersion.PDDLSTREAM,
        **kwargs,
        ) -> bool:
        
        ( 
            robot, joint_ids, 
            obstacles, self_collision_id_pairs, moving_robot_bids, 
            config_type_guard, config_setter, 
            violates_limits, pairwise_link_collision, pairwise_collision
        ) \
        = self._get_arm_collision_params(
            obstacles=obstacles,
            ignore_other_scene_obstacles=ignore_other_scene_obstacles,
            ignore_self_collisions=ignore_self_collisions,
            ignore_gripper_collisions=ignore_gripper_collisions,
            enforce_joint_limits=enforce_joint_limits,
            use_aabb=use_aabb,
            planning_utils_version=planning_utils_version,
            **kwargs
        )
        ATTACHMENTS_DEFAULT = attachments

        @config_type_guard('q', sim=self)
        def arm_collision_fn(q:JointPos, attachments:Iterable[Attachment]=ATTACHMENTS_DEFAULT, 
                             *, return_collision_pair=False, verbose=False):
            assert len(joint_ids) == len(q)
            if violates_limits(q):
                    return True
            with UndoableContext(robot):
                config_setter(q)
                for attachment in attachments:
                    attachment.assign()
                
                # Check for self collisions
                for link1, link2 in iter(self_collision_id_pairs):
                    if pairwise_link_collision(link1, link2):
                        return CollisionPair(link1, link2) if return_collision_pair else True
                # Include attachments as "moving bodies" to be tested for collisions
                attached_bodies = [attachment.child for attachment in attachments]
                moving_body_ids = moving_robot_bids + attached_bodies

                # Check for collisions of the moving bodies and the obstacles
                for moving, obs in product(moving_body_ids, obstacles):
                    if pairwise_collision(moving, obs):
                        return CollisionPair(moving, obs) if return_collision_pair else True
                
                # No collisions detected
                return False
            
        return arm_collision_fn


    def _get_arm_joint_motion_helper_fns(self, 
        obstacles=None,
        *,
        ignore_self_collisions=False,
        ignore_other_scene_obstacles=False,
        ignore_gripper_collisions=False,
        attachments=[],
        enforce_joint_limits=True,
        use_aabb=False, 
        planning_utils_version=PybulletToolsVersion.PDDLSTREAM,
        **kwargs
    ):
        robot_id = self.robot_bid
        joint_ids = self._arm_joint_ids
        pbtools_utils = import_module("pybullet_tools.utils", planning_utils_version)

        get_relevant_kwargs = lambda fn: {k:v for k,v in kwargs.items() if k in ins.signature(fn).parameters}

        distance_fn = pbtools_utils.get_distance_fn(robot_id, joint_ids, **get_relevant_kwargs(pbtools_utils.get_distance_fn))
        sample_fn = pbtools_utils.get_sample_fn(robot_id, joint_ids, **get_relevant_kwargs(pbtools_utils.get_sample_fn))
        extend_fn = pbtools_utils.get_extend_fn(robot_id, joint_ids, **get_relevant_kwargs(pbtools_utils.get_extend_fn))

        collision_fn = self._get_arm_collision_fn(
            obstacles=obstacles,
            ignore_other_scene_obstacles=ignore_other_scene_obstacles,
            ignore_self_collisions=ignore_self_collisions,
            ignore_gripper_collisions=ignore_gripper_collisions,
            attachments=attachments,
            enforce_joint_limits=enforce_joint_limits,
            use_aabb=use_aabb,
            planning_utils_version=planning_utils_version,
            **kwargs,
        )

        return distance_fn, sample_fn, extend_fn, collision_fn
   
    @_convert_args_to_raw_arm_configs('q1','q2')
    def plan_arm_joint_motion(self, 
        q1:JointPos, 
        q2:JointPos, 
        obstacles:Optional[List[BID]]=None,
        algorithm:str='birrt',
        *,
        ignore_other_scene_obstacles=None,
        ignore_self_collisions=False,
        ignore_gripper_collisions=False,

        enforce_joint_limits=False,
        attachments:List[Attachment]=[],
        use_aabb:bool=False, 
        planning_utils_version:PybulletToolsVersion=PybulletToolsVersion.PDDLSTREAM, 
        verbose=False,
        **kwargs
    ):
        if verbose: print(f"\n\nMOTION PLANNING:\n")

        assert len(self._arm_joint_ids) == len(q1) and len(self._arm_joint_ids) == len(q2)

        if obstacles is None:
            obstacles = [self.get_bid(obj) for obj in self._objects if not self.is_movable(obj)]
            # obstacles = [obj for obj in self.env.scene.get_objects() if not self.is_movable(obj)]
        if verbose: print(f"MP Obstacles: {[self.get_name(obj) for obj in obstacles]}")

        distance_fn, sample_fn, extend_fn, collision_fn = self._get_arm_joint_motion_helper_fns( 
            obstacles=obstacles,
            attachments=attachments,
            ignore_other_scene_obstacles=ignore_other_scene_obstacles,
            ignore_self_collisions=ignore_self_collisions,
            ignore_gripper_collisions=ignore_gripper_collisions,
            enforce_joint_limits=enforce_joint_limits,
            use_aabb=use_aabb,
            planning_utils_version=planning_utils_version,
            **kwargs
        )
        with UndoableContext(self._robot):
            if (collision := collision_fn(q1, return_collision_pair=True)) is not False:
                if verbose: print(f"Initial configuration in collision:\nCollision Links: {collision}\nConfig q1={q1}")
                plan = None
            elif (collision := collision_fn(q2, return_collision_pair=True)) is not False:
                if verbose: print(f"End configuration in collision:\nCollision Links: {collision}\nConfig q2={q2}")
                plan = None
            else:
                if verbose: print(f"Beginning motion planning with algorithm {algorithm}...")
                mp_algo_fn = self._get_motion_planning_algorithm(algorithm)
                if algorithm == 'direct':
                    plan = mp_algo_fn(q1, q2, extend_fn, collision_fn)
                elif algorithm == 'birrt':
                    plan = mp_algo_fn(q1, q2, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs)
                elif algorithm == 'rrt_star':
                    plan = mp_algo_fn(q1, q2, distance_fn, sample_fn, extend_fn, collision_fn, max_iterations=5000, **kwargs)
                elif algorithm == 'rrt':
                    plan = mp_algo_fn(q1, q2, distance_fn, sample_fn, extend_fn, collision_fn, iterations=500, **kwargs)
                elif algorithm == 'lazy_prm':
                    plan = mp_algo_fn(q1, q2, distance_fn, sample_fn, extend_fn, collision_fn, [500, 2000, 5000], **kwargs)
                else:
                    plan = None
            
            if verbose: print(f"\nMotion planning complete. Plan: {plan}\n\n")

            return plan



    def get_grasp_gen(self):
        grasps = super().get_grasp_gen()
        def gen(body:UID):
            grasp_gen= grasps(body)
            for grasp, *_ in grasp_gen:
                out = (grasp, BodyPose(self.robot_bid, grasp.approach_pose), BodyPose(self.robot_bid, grasp.grasp_pose))
                yield out
        return gen

    def get_grasp_traj_fn(self, num_attempts: int = 10) -> Callable[[str, BodyGrasp], Tuple[JointPos, Command]]:
        calculate_grasp_command = super().get_grasp_traj_fn(num_attempts)
        @ft.wraps(calculate_grasp_command)
        def wrapper(target:UID, grasp:BodyGrasp):
            try:
                out = calculate_grasp_command(target, grasp)
                if out is not None:
                    q, cmd = out
                    conf = BodyConf(body=self.robot_bid, joints=self._arm_joint_ids, configuration=q)
                    command = ArmCommand(
                        body_paths=[
                            ArmPath(self, path.path, path.attachments) 
                            if isinstance(path,BodyPath) 
                            else path for path in cmd.body_paths
                    ])
                    out = (conf, command)
                else:
                    out = ()
                return out
            except (TypeError, AssertionError):
                return None
        return wrapper
    
    def arm_fk(self, q):
        pose = super().arm_fk(q)
        return (BodyPose(self.robot_bid, pose),) if pose is not None else None
    
    def arm_ik(self, *args, **kwargs):
        q = super().arm_ik(*args, **kwargs)
        return (ArmConf(self, q),) if q is not None else None




        


class ArmConf(BodyConf):
    def __init__(self, sim:ModifiedMiGSI, configuration:JointPos):
        super().__init__(body=sim.robot_bid, joints=sim._arm_joint_ids, configuration=configuration)
        self.sim = sim

# class ArmConf(BodyConf):
#     def __init__(self, sim:ModifiedMiGSI, q:JointPos):
#         if isinstance(q, BodyConf):
#             assert q.body == sim.robot_bid and q.joints == sim._arm_joint_ids
#             q = q.configuration
#         super().__init__(sim.robot_bid, q, sim._arm_joint_ids)
#         self.sim = sim
#     def eef_pose(self):
#         return self.sim.arm_fk(self.configuration)


class ArmCommand(Command):

    def control(self, real_time=False, dt=0.05, verbose=False):
        for body_path in self.body_paths:
            body_path.control(real_time=real_time, dt=dt) #, verbose=verbose)

    def execute(self, time_step=0.05, step_sim=False):
        for i, body_path in enumerate(self.body_paths):
            if isinstance(body_path, ApplyForce):
                body_path.control()
            else:
                for q in body_path.iterator(step_sim=step_sim):
                    pass# wait_for_duration(time_step)



class ArmPath(BodyPath):
    def __init__(self, sim:ModifiedMiGSI, path:Iterable[JointPos], attachments:List[Attachment]=[]):
        super().__init__(sim.robot_bid, path, sim._arm_joint_ids, attachments)
        self.sim = sim

    # def executor(self): # more accurate name
    def iterator(self, step_sim=False):
        for configuration in self.path:
            if step_sim:
                self.sim.set_arm_config(configuration, self.attachments) #, _sync_viewer=False)
                self.sim.step()
            else:
                self.sim.set_arm_config(configuration, self.attachments)
            yield configuration

    def refine(self, num_steps=0, update=False):
        refined_path = self.__class__(self.sim, refine_path(self.body, self.joints, self.path, num_steps), self.attachments)
        if update:
            self.path = refined_path
        return refined_path
    
    def control(self, real_time=False, dt=0.05, verbose=False):
        for configuration in self.path:
            action = np.zeros(self.sim.env.action_space.shape)
            action[4:12] = configuration

            q = self.sim.get_arm_config()
            while not np.allclose(q, configuration, atol=1e-3, rtol=0):
                state, reward, done, x = self.sim.step(action)
                q = self.sim.get_arm_config()
            # wait_for_duration(dt)

            # if verbose:
            #     print('Action:', [x for x in action])
            #     print('Config:', self.sim.get_arm_config())
            #     print()
    
    def __repr__(self):
        if self.path is None:
            return f"{self.__class__.__name__}(None)"
        else:
            return super().__repr__() 
    




def init_sim(objects, headless=True, **kwargs) -> ModifiedMiGSI:
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = "fetch_tamp.yaml"
    config_path = os.path.join(dir_path,config)

    robot_pose=((-1.2,5.4,0),(0,0,pi)) 
    # viewer_pose=((-1.8,0.8,1.8),(0.1,0.8,-0.6))
    viewer_pose=((-2.0,4.5,1.8),(0.1,0.6,-0.8))
    # viewer_pose=((-0.6,3.9,1.8),(-0.4,0.9,-0.3))

    KWARGS = {
        'headless' : headless,
        'robot_pose' : robot_pose,
        'viewer_pose' : viewer_pose,
        # 'load_object_categories': ['sink','stove', 'bottom_cabinet'],
        # 'load_room_types' : ['kitchen'],
        'action_timestep' : 1 / 30.0,
        'physics_timestep' : 1 / 120.0,
    }
    KWARGS.update(kwargs)

    sim = ModifiedMiGSI(
        config_file=config_path,
        objects=objects,
        **KWARGS
    )

    return sim

##############################################################################






def motion_plan(sim:ModifiedMiGSI, q1, q2, o=None, g=None, as_tuple=True, **kwargs):
    # if isinstance(q1,BodyConf): q1 = q1.configuration
    # if isinstance(q2,BodyConf): q2 = q2.configuration
    if o is not None:
        assert g is not None and g.body==sim.get_name(o)
        attachments = [g.attachment()]
    else:
        attachments=[]
    path = sim.plan_arm_joint_motion(q1, q2, attachments=attachments, **kwargs)
    if path is None:
        return None
    path = ArmPath(sim, path, attachments)
    return (path,) if as_tuple else path

def plan_grasp_and_carry(sim:ModifiedMiGSI, q1, q2, obj, g, as_tuple=True, **kwargs):
    # if isinstance(q1,BodyConf): q1 = q1.configuration
    # if isinstance(q2,BodyConf): q2 = q2.configuration
    # grasp = next(sim.get_grasp_gen()(obj))[0]
    attachments = [g.attachment()]
    path = sim.plan_arm_joint_motion(q1, q2, attachments=attachments, **kwargs)
    if path is None:
        return None
    path = ArmPath(sim, path, attachments)    
    return (path,) if as_tuple else path


def print_fluent(sim:ModifiedMiGSI, fluent:tuple, dec:int=3, return_only=False):
    assert isinstance(fluent, tuple)
    formatted_fluent = tuple(
        (sim.get_name(arg) if isinstance(arg,int) else 
        tuple(np.round(arg.configuration,dec)) if isinstance(arg,BodyConf) else 
        (sim.get_name(arg.body), *(tuple(np.round(p,dec)) for p in arg.pose)) if isinstance(arg, BodyPose) else 
        arg) for arg in fluent
    )
    if not return_only:
        print(formatted_fluent)
    return formatted_fluent

def get_pddlproblem(sim:ModifiedMiGSI, q_init:Optional[Union[float,BodyConf]], q_goal:Union[float,BodyConf], verbose=True):
    if q_init is None:                     
        q_init = ArmConf(sim, sim.get_arm_config())
    else:
        sim.set_arm_config(q_init)
        if not isinstance(q_init, ArmConf): 
            q_init = ArmConf(sim, q_init)
    if not isinstance(q_goal, ArmConf): 
        q_goal = ArmConf(sim, q_goal)

    if verbose:
        print()
        print('Initial configuration: ', q_init.configuration)
        print('Goal configuration: ', q_goal.configuration)
        print()

    # objects  = [sim.get_bid(obj) for obj in sim.objects]

    _subjects = [q_init, q_goal, *sim.objects]
    
    init = [
        ('AtConf', q_init),
    ]
    for sbj in _subjects:
        if isinstance(sbj,BodyConf):
            init += [('Conf', sbj)]
        elif sbj in sim.objects:            
            ObjType = sim.get_object(sbj).category.capitalize()
            init += [('Obj', sbj), (ObjType, sbj)]
            if sim.is_movable(sbj):
                pose = BodyPose(sbj, [tuple(x) for x in sim.get_pose(sbj)]) # cast back from np.array for less visual clutter
                init +=[
                    ('Pose', pose),
                    ('AtPose', sbj, pose),
                    ('Movable', sbj)
                ]
                # find surface that it is on
                for _obj in sim.objects:
                    if not sim.is_movable(_obj) and not _obj in ['walls', 'ceilings']:
                        if sim.is_on(sbj, _obj):
                            init += [('Placement', pose, sbj, _obj)]

    goal = []
    goal += [('AtConf', q_goal)]
    # goal += [('Holding', 'radish')]
    # goal += [('On', 'celery', 'stove')]
    # goal += [('Cooked', 'radish')]
    assert isinstance(goal, list)
    if len(goal) == 0:
        raise ValueError("No goal specified.")
    elif len(goal) == 1:
        goal, *_ = goal
    else:
        goal = ('and', *goal)


    if verbose:
        print(f"\n\nINITIAL STATE")
        for fluent in init:
            print_fluent(sim, fluent)
            # fluent = tuple(
            #     (sim.get_name(arg) if isinstance(arg,int) else 
            #     tuple(np.round(arg.configuration,3)) if isinstance(arg,BodyConf) else 
            #     (sim.get_name(arg.body), *(tuple(np.round(p,3)) for p in arg.pose)) if isinstance(arg, BodyPose) else 
            #     arg) for arg in fluent
            # )
            # print(fluent)
        print('\n')
        print("GOAL STATE")
        if isinstance(goal, list):
            for fluent in goal:
                if fluent != "and":
                    print_fluent(sim, fluent)
        elif isinstance(goal, tuple):
            print_fluent(sim, goal)
        else:
            print(goal)
        print('\n')
        


    domain_pddl = read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'toy_domain.pddl'))
    constant_map = {}
    stream_pddl =  read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'toy_stream.pddl'))
    
    sample_arm_config = sim.sample_arm_config
    motion_planner = ft.partial(motion_plan, sim, as_tuple=True)
    carry_planner = ft.partial(plan_grasp_and_carry, sim, as_tuple=True)
    stream_map = {
        'sample-conf' : from_fn(sample_arm_config),
        'is-surface' : from_test(sim.is_surface),
        'sample-placement' : from_gen_fn(sim.get_stable_gen()),
        'sample-grasp' : from_gen_fn(sim.get_grasp_gen()),
        'grasp-command' : from_fn(sim.get_grasp_traj_fn()),
        'motion-plan' : from_fn(motion_planner),
        # 'motion-plan-carry' : from_fn(carry_planner),
        'forward-kinematics' : from_fn(sim.arm_fk),
        'inverse-kinematics' : from_fn(sim.arm_ik),
    }


    problem=PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)
    return problem


def run_planner(sim:ModifiedMiGSI, 
                qinit:Optional[Union[float,BodyConf]]=None, 
                qgoal:Optional[Union[float,BodyConf]]=None, 
                max_iterations:int=20,
                debug:bool=False, 
                print_results:bool=True):
    if qgoal is None: 
        qgoal = sim._robot.tucked_default_joint_pos[sim._arm_joint_control_indices]

    problem = get_pddlproblem(sim, qinit, qgoal)
    solution = solve(problem, unit_costs=True, verbose=debug, debug=debug, max_iterations=max_iterations)

    if print_results:
        print(solution.plan)
        print('\n\n')
        for fluent in solution.certificate.all_facts:
            fluent = tuple(
                (sim.get_name(arg) if isinstance(arg,int) else 
                tuple(np.round(arg.configuration,3)) if isinstance(arg,BodyConf) else 
                (sim.get_name(arg.body), *(tuple(np.round(p,3)) for p in arg.pose)) if isinstance(arg, BodyPose) else 
                arg) for arg in fluent
            )

            print(fluent)

    return solution

def consolidate_plan(plan):
    paths = []
    for name, args in plan:
        if name == 'place':
            paths += args[-1].reverse().body_paths
        elif name in ['move', 'move_free', 'move_holding', 'pick']:
            action = args[-1]
            if isinstance(action, Command):
                paths += action.body_paths
            elif isinstance(action, ArmPath):
                paths += [action]
            else:
                raise NotImplementedError(f"no handling implemented for action ({name} {args}) with args[-1] of type {type(args[-1])}")
    return ArmCommand(paths)

def visualize_sim(sim:ModifiedMiGSI, debug=False):
    sim.env.simulator.viewer.reset_viewer()

    if debug:
        loop_and_report_fps(sim.step)
    else:
        while True:
            # sim.env.simulator.viewer.update()
            sim.step()

            # sim.env.simulator.sync()
            # action = np.zeros(sim.env.action_space.shape)
            # state, reward, done, x = sim.env.step(action)
            # cv2.setWindowProperty("RobotView", cv2.WND_PROP_TOPMOST, 1)
            # cv2.setWindowProperty("Viewer", cv2.WND_PROP_TOPMOST, 1)
            # print(state, reward, done, x)
            # print(sim.get_pose('stove'))
            # print(sim.get_object('stove').states[object_states.AABB].get_value())


def get_objects():
    counter = ObjSpec("counter", "countertop", model='counter_0',
                      position=(-2.35,  4.95,  0.35),
                      orientation=(0,0,pi/2),
                      fixed_base=True,
                      bounding_box=(1.0, 0.5, 0.75),
                      )
    sink   = ObjSpec("sink",  "sink", model='kitchen_sink',    
                     position=(-2.35,  5.8,  0.3),
                     orientation=(0,0,pi/2),
                     fixed_base=True,
                     bounding_box=(0.75,  0.5,  0.75),
                     )
    stove  = ObjSpec("stove",  "stove", model='101924',
                     position=(-0.225  , 5.75,  0.6), 
                     orientation=(0,0,-pi/2),
                     fixed_base=True,
                     bounding_box=(1.0,  0.75,  1.25),
                    )
    celery = ObjSpec("celery", "celery", model='celery_000', 
                     position=(-2.3,  5.1,  1.0), place_on='counter', fixed_base=False, scale=0.15)
    radish = ObjSpec("radish", "radish", model='45_0', 
                     position=(-2.15,  5.3,  1.0), place_on='counter', fixed_base=False, scale=1)
    
    return sink, stove, counter, celery, radish

class MyFetch(Fetch):

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        # We use multi finger gripper, differential drive, and IK controllers as default
        controllers["base"] = "DifferentialDriveController"
        controllers["camera"] = "JointController"
        controllers["arm_{}".format(self.default_arm)] = "JointController"
        controllers["gripper_{}".format(self.default_arm)] = "MultiFingerGripperController"

        return controllers

    @property
    def _default_controller_config(self):
        # Grab defaults from super method first
        cfg = super()._default_controller_config

        # Use default IK controller -- also need to override joint idx being controlled to include trunk in default
        # IK arm controller
        cfg["arm_{}".format(self.default_arm)]["JointController"]["joint_idx"] = np.concatenate(
            [self.trunk_control_idx, self.arm_control_idx[self.default_arm]]
        )

        # If using rigid trunk, we also clamp its limits
        if self.rigid_trunk:
            cfg["arm_{}".format(self.default_arm)]["JointController"]["control_limits"]["position"][0][
                self.trunk_control_idx
            ] = self.untucked_default_joint_pos[self.trunk_control_idx]
            cfg["arm_{}".format(self.default_arm)]["JointController"]["control_limits"]["position"][1][
                self.trunk_control_idx
            ] = self.untucked_default_joint_pos[self.trunk_control_idx]

        return cfg



def loop_and_report_fps(f:Callable, avg=False):
    import time
    if avg:
        while True:
            interval = 1
            frames = 0
            start = time.process_time()
            now = start
            end = start + interval
            while now < end:
                f()
                now = time.process_time()
                frames += 1
            print(f"FPS: {frames/(now-start)}, frames: {frames}, t: {now-start}")
    else:
        while True:
            start = time.process_time()
            f()
            end = time.process_time()
            print(f"FPS: {1/(end-start)}, t: {end-start}")


def main():
    args = commandline_args()
    GUI = decide_gui(args)
    DEBUG = False
    USE_CONTROLLERS = False
    REFINE_PATH = False
    
    sim = init_sim(
        objects=get_objects(), 
        headless=(not GUI), 
        tensor=True,
        let_settle=True,
        # viewer_pose=([-2.2, 4.8, 1.5], [0.6, 0.6, -0.4]),
        visualize_planning=True,
    )

    try:
        # if GUI:
        #     visualize_sim(sim, debug=DEBUG)
        # else:
        #     loop_and_report_fps(sim.step)
        # sys.exit()


        qinit = sim._robot.untucked_default_joint_pos[sim._arm_joint_control_indices]
        qgoal = sim._robot.tucked_default_joint_pos[sim._arm_joint_control_indices]


        sim.step()
        solution = run_planner(sim, 
                               qinit=qinit,
                               qgoal=qgoal,
                               debug=DEBUG, 
                               print_results=False
                            )
        plan = solution.plan
        
        

        if plan is None:
            print("\nNo solution found. Exiting...")
        else:
            print("\n#####################################################\nSolution:")
            print(plan, "\n")
            if GUI:
                command = consolidate_plan(plan)
                
                if USE_CONTROLLERS:
                    wait_for_user('Simulate?')
                    command.control()
                else:
                    wait_for_user('Execute?')
                    dt = sim.env.simulator.render_timestep
                    if REFINE_PATH:
                        refinement_factor = 10
                        dt /= refinement_factor
                        command = command.refine(num_steps=refinement_factor)
                    command.execute(time_step=dt, step_sim=True)

                print("Config: ", sim.get_arm_config())
                # wait_for_user('Show qgoal?')
                # sim.set_arm_config(qgoal, freeze=True)
                # print("Config: ", sim.get_arm_config())
                # sim.sync()
                print()
                print("Press 'Ctrl+C' in terminal or 'esc' in Viewer window to exit.")

                while True:
                    sim.step()

    finally:
        sim.env.close()

    
    


def commandline_args():
    parser = create_parser()
    parser.add_argument('-g','--gui', action='store_true', help='Render and visualize the system')
    # parser.add_argument('--vgl', action='store_true', help='Enable VirtualGL (for remote rendering)')
    args = parser.parse_args()
    return args


def decide_gui(args:Namespace): # gui:bool=None, vgl:bool=None):
    # defer to commandline arg, then to environmental variable
    args.gui = args.gui if args.gui else bool(os.getenv("GUI"))
    if args.gui:
            print("Visualization on.")
    return args.gui






if __name__=='__main__':
    main()