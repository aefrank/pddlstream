from collections import UserDict
from itertools import product, takewhile
from json import load
import os
import functools as ft
import inspect as ins
from time import sleep
import cv2
import sys
from typing_extensions import TypeAlias
from igibson import object_states
from igibson.envs.igibson_env import iGibsonEnv
from igibson.robots import Fetch
import numpy as np
from typing import Any, Callable, List, NamedTuple, NamedTuple, NewType, NoReturn, Optional, Tuple, Iterable, Union

from igibson.utils.assets_utils import get_all_object_categories, get_ig_model_path, get_object_models_of_category
from numpy import pi, random
from transforms3d.euler import quat2euler
from examples.fetch.from_kuka_tamp.utils.utils import PybulletToolsVersion, import_module
from iGibson.igibson.utils.assets_utils import get_ig_avg_category_specs

from igibson.action_primitives.starter_semantic_action_primitives import URDFObject, UndoableContext
from igibson.objects.object_base import BaseObject

from examples.fetch.from_kuka_tamp.fetch_primitives import BID, UID, JointPos, KinematicConstraint, MyiGibsonSemanticInterface, Object, _sync_viewer_after_exec, iGibsonSemanticInterface
from examples.fetch.from_kuka_tamp.utils.object_spec import Euler, Kwargs, ObjectSpec, Orientation3D, Position3D, Quaternion
from examples.pybullet.utils.pybullet_tools.kuka_primitives import ApplyForce, Attach, BodyConf, BodyGrasp, BodyPose, BodyPath, Command
from pddlstream.algorithms.meta import create_parser, solve
from pddlstream.language.constants import PDDLProblem
from pddlstream.language.generator import from_fn, from_gen_fn, from_test
from pddlstream.utils import Profiler, read

from igibson.external.pybullet_tools.utils import refine_path, wait_for_user
from examples.pybullet.utils.pybullet_tools.utils import Attachment, wait_for_duration


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
        self.data = {'name':name, 'category':category}
        
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

    def __init__(self, config_file:str, objects:Iterable[ObjSpec]=[], headless:bool=True, let_settle:bool=False,
                 *, robot_pose:Pose=((0,0,0),(0,0,0)), viewer_pose:Optional[Pose]=None, verbose=True, **config_options:Kwargs):
        self.has_gui = not headless
        self._init_igibson(config=config_file, headless=headless, **config_options)
        self._init_objects(objects)
        self._init_robot_state(*robot_pose)
        if not headless:
            self._viewer_setup(*viewer_pose) if viewer_pose is not None else self._viewer_setup()
        if let_settle:
            print("Allowing to settle... ", end="")
            for _ in range(100):
                self.step()
            print("done.")


    def _load_objects(self, specs:Iterable[ObjSpec], verbose: bool = True) -> List[BID]:
        def load_single_object(spec:ObjSpec):
            if verbose: print(f"Loading {spec['category'].capitalize()} object '{spec['name']}'...", end='')  

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
            self.land(obj, position, orientation, _sync_viewer=False)
            # print(position)

            assert obj.states[object_states.OnTop].get_value(surface) #, use_ray_casting_method=True)
        else:
            self.set_position_orientation(name, position, orientation, _sync_viewer=False)
        if not state=={}:
            raise NotImplementedError(f"No handling defined for state variables {list(state)}")
    
    def _init_objects(self, specs:Iterable[ObjSpec]) -> None:
        self._load_objects(specs)
        
        pose = self.get_pose(self.robot)
        self.set_position(self.robot, (100,100,100), _sync_viewer=False)
        for spec in specs:
            self._init_object_statevars(spec.name, **spec.state)
        self.set_pose(self.robot, pose, _sync_viewer=False)
    
    def _init_object_state(self, obj_specs: Iterable[ObjectSpec], *, verbose=True) -> None:
        raise NotImplementedError


    def step(self, *, action:np.ndarray=None):
        if action is not None:
            self.env.step(action)
        else:
            self.env.simulator.step()

    def is_surface(self, obj:UID):
        return not self.is_movable(obj) and (self.get_object(obj).category.lower() not in ['walls', 'ceilings'])
    
    @_sync_viewer_after_exec
    def land(self, obj:Object, position:Optional[Position3D]=None, orientation:Optional[Orientation3D]=None):
        if position is None:    position    = self.get_position(obj)
        if orientation is None: orientation = self.get_orientation(obj)
        if len(orientation)==4: orientation = quat2euler(orientation)

        obj = self.get_object(obj)
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
        config_setter = ft.partial(self.set_arm_config, _sync_viewer=False)
        pbtools_utils = import_module("pybullet_tools.utils", planning_utils_version)

        relevant_kwargs = get_relevant_kwargs(pbtools_utils.all_between)
        violates_limits = lambda q: pbtools_utils.all_between(lower_limits, q, upper_limits, **relevant_kwargs)

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
            use_aabb=use_aabb,
            planning_utils_version=planning_utils_version,
            **kwargs
        )
        ATTACHMENTS_DEFAULT = attachments

        @config_type_guard('q', sim=self)
        def arm_collision_fn(q:JointPos, attachments:Iterable[Attachment]=ATTACHMENTS_DEFAULT, *, verbose=False):
            assert len(joint_ids) == len(q)
            with UndoableContext(robot):
                if violates_limits(q):
                    return True
                config_setter(q)
                for attachment in attachments:
                    attachment.assign()
                
                # Check for self collisions
                for link1, link2 in iter(self_collision_id_pairs):
                    if pairwise_link_collision(link1, link2):
                        return True
                # Include attachments as "moving bodies" to be tested for collisions
                attached_bodies = [attachment.child for attachment in attachments]
                moving_body_ids = moving_robot_bids + attached_bodies

                # Check for collisions of the moving bodies and the obstacles
                for moving, obs in product(moving_body_ids, obstacles):
                    if pairwise_collision(moving, obs):
                        return True
                
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
        use_aabb=False, 
        planning_utils_version=PybulletToolsVersion.PDDLSTREAM,
        **kwargs
    ):
        robot_id = self.robot_bid
        joint_ids = self._arm_joint_ids
        pbtools_utils = import_module("pybullet_tools.utils", planning_utils_version)

        get_relevant_kwargs = lambda fn: {k:v for k,v in kwargs.items() if k in ins.signature(fn).parameters}
        check_initial_end = ft.partial(pbtools_utils.check_initial_end, **get_relevant_kwargs(pbtools_utils.check_initial_end))
        distance_fn = pbtools_utils.get_distance_fn(robot_id, joint_ids, **get_relevant_kwargs(pbtools_utils.get_distance_fn))
        sample_fn = pbtools_utils.get_sample_fn(robot_id, joint_ids, **get_relevant_kwargs(pbtools_utils.get_sample_fn))
        extend_fn = pbtools_utils.get_extend_fn(robot_id, joint_ids, **get_relevant_kwargs(pbtools_utils.get_extend_fn))

        collision_fn = self._get_arm_collision_fn(
            obstacles=obstacles,
            ignore_other_scene_obstacles=ignore_other_scene_obstacles,
            ignore_self_collisions=ignore_self_collisions,
            ignore_gripper_collisions=ignore_gripper_collisions,
            attachments=attachments,
            use_aabb=use_aabb,
            planning_utils_version=planning_utils_version,
            **kwargs,
        )

        return check_initial_end, distance_fn, sample_fn, extend_fn, collision_fn
   
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
        attachments:List[Attachment]=[],
        planning_utils_version:PybulletToolsVersion=PybulletToolsVersion.PDDLSTREAM, 
        use_aabb:bool=False, 
        **kwargs
    ):
        # if isinstance(q1,BodyConf): q1 = q1.configuration
        # if isinstance(q2,BodyConf): q2 = q2.configuration

        assert len(self._arm_joint_ids) == len(q1) and len(self._arm_joint_ids) == len(q2)

        # if obstacles is None:
        #     # obstacles = [obj for obj in self._objects if not self.is_movable(obj)]
        #     obstacles = [obj for obj in self.env.scene.get_objects() if not self.is_movable(obj)]

        check_initial_end, distance_fn, sample_fn, extend_fn, collision_fn = self._get_arm_joint_motion_helper_fns( 
            obstacles=obstacles,
            attachments=attachments,
            ignore_other_scene_obstacles=ignore_other_scene_obstacles,
            ignore_self_collisions=ignore_self_collisions,
            ignore_gripper_collisions=ignore_gripper_collisions,
            planning_utils_version=planning_utils_version,
            use_aabb=use_aabb,
            **kwargs
        )
        with UndoableContext(self._robot):
            if not check_initial_end(q1, q2, collision_fn):
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
                return out
            except (TypeError, AssertionError):
                return None
        return wrapper
    
    def arm_fk(self, q):
        pose = super().arm_fk(q)
        return (BodyPose(self.robot_bid, pose),)



        


class ArmConf(BodyConf):
    def __init__(self, sim:ModifiedMiGSI, configuration:JointPos):
        super().__init__(body=sim.robot_bid, joints=sim._arm_joint_ids, configuration=configuration)
        self.sim = sim


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
                    wait_for_duration(time_step)

class ArmPath(BodyPath):
    def __init__(self, sim:ModifiedMiGSI, path:Iterable[JointPos], attachments:List[Attachment]=[]):
        super().__init__(sim.robot_bid, path, sim._arm_joint_ids, attachments)
        self.sim = sim

    # def executor(self): # more accurate name
    def iterator(self, step_sim=False):
        for configuration in self.path:
            if step_sim:
                self.sim.set_arm_config(configuration, self.attachments, _sync_viewer=False)
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
    viewer_pose=((-0.6,3.9,1.8),(-0.4,0.9,-0.3))

    KWARGS = {
        'headless' : headless,
        'robot_pose' : robot_pose,
        'viewer_pose' : viewer_pose,
        # 'load_object_categories': ['sink','stove', 'bottom_cabinet'],
        # 'load_room_types' : ['kitchen'],
    }
    KWARGS.update(kwargs)

    sim = ModifiedMiGSI(
        config_file=config_path,
        objects=objects,
        **KWARGS
    )

    return sim

##############################################################################


def motion_plan(sim:ModifiedMiGSI, q1, q2, o=None, g=None, as_tuple=False):
    if isinstance(q1,BodyConf): q1 = q1.configuration
    if isinstance(q2,BodyConf): q2 = q2.configuration
    if o is not None:
        assert g is not None and g.body==sim.get_name(o)
        attachments = [g.attachment()]
    else:
        attachments=[]
    path = ArmPath(sim, sim.plan_arm_joint_motion(q1,q2,attachments=attachments), attachments)
    return path if not as_tuple else (path,)

def plan_grasp_and_carry(sim:ModifiedMiGSI, q1, q2, obj, g, as_tuple=False):
    if isinstance(q1,BodyConf): q1 = q1.configuration
    if isinstance(q2,BodyConf): q2 = q2.configuration
    # grasp = next(sim.get_grasp_gen()(obj))[0]
    attachments = [g.attachment()]
    path = ArmPath(sim, sim.plan_arm_joint_motion(q1,q2,attachments=attachments), attachments)
    return path if not as_tuple else (path,)



def get_pddlproblem(sim:ModifiedMiGSI, q_init:Optional[Union[float,BodyConf]], q_goal:Union[float,BodyConf], verbose=True):
    if q_init is None:                     
        q_init = BodyConf(sim.robot_bid, sim.get_arm_config())
    else:
        sim.set_arm_config(q_init)
        if not isinstance(q_init, BodyConf): 
            q_init = BodyConf(sim.robot_bid, q_init)
    if not isinstance(q_goal, BodyConf): 
        q_goal = BodyConf(sim.robot_bid, q_goal)

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


    # goal = ('AtConf', q_goal)
    goal = ('Holding', 'radish')
    # goal = ('On', 'celery', 'stove')
    # goal = ('Cooked', 'radish')

    if verbose:
        print(f"\n\nINITIAL STATE")
        for fluent in init:
            fluent = tuple(
                (sim.get_name(arg) if isinstance(arg,int) else 
                tuple(np.round(arg.configuration,3)) if isinstance(arg,BodyConf) else 
                (sim.get_name(arg.body), *(tuple(np.round(p,3)) for p in arg.pose)) if isinstance(arg, BodyPose) else 
                arg) for arg in fluent
            )

            print(fluent)
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
                debug:bool=False, 
                print_results:bool=True):
    if qgoal is None: 
        qgoal = sim._robot.tucked_default_joint_pos[sim._arm_joint_control_indices]

    problem = get_pddlproblem(sim, qinit, qgoal)
    solution = solve(problem, unit_costs=True, verbose=debug, debug=debug)

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

def visualize_sim(sim:ModifiedMiGSI):
    sim.env.simulator.viewer.reset_viewer()

    while True:
        sim.env.simulator.sync()
        action = np.zeros(sim.env.action_space.shape)
        state, reward, done, x = sim.env.step(action)
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


def decide_gui(*, force:bool=None):
    if force is not None:
        return force
    
    parser = create_parser()
    parser.add_argument('-g','--gui', action='store_true', help='Render and visualize the system')
    args = parser.parse_args()

    gui = args.gui if args.gui else bool(os.getenv("GUI"))
    if gui:
        print("Visualization on.")
    
    return gui



def main():
    GUI = decide_gui()
    DEBUG = False
    USE_CONTROLLERS = False
    REFINE_PATH = False
    
    sim = init_sim(
        objects=get_objects(), 
        headless=(not GUI), 
        let_settle=True,
        viewer_pose=([-2.2, 4.8, 1.5], [0.6, 0.6, -0.4]),
        visualize_planning=True,
    )

    try:
        qinit = sim._robot.tucked_default_joint_pos[sim._arm_joint_control_indices]
        # qgoal = sim._robot.untucked_default_joint_pos[sim._arm_joint_control_indices]
        qgoal = sim.as_arm_config(sim.get_position('radish'))

        solution = run_planner(sim, 
                               qinit=qinit,
                               qgoal=qgoal,
                               debug=DEBUG, 
                               print_results=False
                            )
        plan = solution.plan
        

        # if GUI:
        #     visualize_sim(sim)

        if plan is None:
            print("\nNo solution found. Exiting...")
        else:
            print("\n#####################################################\nSolution:")
            print(plan, "\n")
            if GUI:
                command = consolidate_plan(plan)
                # path = plan[0].args[-1].path
                
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
                wait_for_user('Show qgoal?')
                sim.set_arm_config(qgoal, freeze=True)
                print("Config: ", sim.get_arm_config())
                sim.step()
                print()
                print("Press 'Ctrl+C' in terminal or 'esc' in Viewer window to exit.")

                while True:
                    sim.step()

    finally:
        sim.env.close()

    
    

    







if __name__=='__main__':
    main()