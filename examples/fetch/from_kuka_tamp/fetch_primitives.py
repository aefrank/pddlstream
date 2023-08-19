from collections import UserDict
import functools as ft
import os
import sys
from typing import Any, Generator, List, Optional, Tuple, Union
from igibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
from igibson.action_primitives.starter_semantic_action_primitives import MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE, PREDICATE_SAMPLING_Z_OFFSET, UndoableContext

from igibson import object_states
from igibson.external.pybullet_tools.utils import is_collision_free
from igibson.object_states.utils import sample_kinematics
from igibson.objects.object_base import BaseObject
from igibson.envs.igibson_env import iGibsonEnv
from igibson.robots import BaseRobot
from igibson.scenes.igibson_indoor_scene import URDFObject
from igibson.utils.assets_utils import get_ig_avg_category_specs
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
from igibson.utils.utils import parse_config, quatToXYZW
from transforms3d.euler import euler2quat

from examples.fetch.from_kuka_tamp.object_spec import (
    ObjectSpec, 
    URDFObjectSpec,
    Orientation, 
    Position,
    UniqueID, 
)


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
    
    @property
    def robot(self) -> UniqueID:
        assert len(self.robots)==1
        return self.robots[0]
    
    @property
    def _robot(self) -> BaseRobot:
        assert len(self._robots)==1
        return self._robots[0]

    ################################################################################

    def __init__(self, config_file:str, objects:List[ObjectSpec], *, headless:bool=True, verbose=True):
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

    def load_objects(self, specifications:List[Union[str,dict]], verbose:bool=True) -> None:
        '''Load objects into simulator based on a list of categories.
        ''' 
        body_ids_by_object = {}
        for spec in specifications:
            spec = _as_urdf_spec(spec)
            body_ids = self.load_object(spec=spec, verbose=verbose)
            body_ids_by_object[spec["name"]] = body_ids 
        return body_ids_by_object
    
    def _init_object_state(self, obj_specs:List[ObjectSpec], *, verbose=True) -> None:
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
                body2 = link_b_list = None

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
    
    ################################################################################
    
    def get_stable_gen(self): # -> Generator[Tuple[Position, Orientation], None, None]:
        def gen(body:UniqueID, surface:UniqueID):
            while True:
                placement = self.sample_placement(body, surface)
                if placement is not None:
                    yield placement
        return gen
    
    def get_grasp_gen(self, robot, grasp_name='top'):
        pass

    def get_tool_link(self, robot):
        pass

    def get_ik_fn(self, robot, fixed=[], teleport=False, num_attempts=10):
        pass

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


    
    


# def sample_placement(body:UniqueID, surface:UniqueID):
#     placement = sample_kinematics(predicate=object_states.OnTop,
#                         objA=body,
#                         objB=surface,
#                         use_ray_casting_method=True,
#                         max_trials=MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE,
#                         skip_falling=True,
#                         z_offset=PREDICATE_SAMPLING_Z_OFFSET,
#     )
#     return placement

# def sample_kinematics(
#     predicate,
#     objA,
#     objB,
#     use_ray_casting_method=False,
#     max_trials=100,
#     z_offset=0.05,
#     skip_falling=False,
# ):
#     '''Modified version of igibson/object_states/utils.py
#     '''

#     sample_on_floor = predicate == "onFloor"

#     if not use_ray_casting_method and not sample_on_floor and predicate not in objB.supporting_surfaces:
#         return False

#     objA.force_wakeup()
#     if not sample_on_floor:
#         objB.force_wakeup()

#     state_id = pb.saveState()
#     for _ in range(max_trials):
#         pos = None
#         if hasattr(objA, "orientations") and objA.orientations is not None:
#             orientation = objA.sample_orientation()
#         else:
#             orientation = [0, 0, 0, 1]

#         # Orientation needs to be set for stable_z_on_aabb to work correctly
#         # Position needs to be set to be very far away because the object's
#         # original position might be blocking rays (use_ray_casting_method=True)
#         old_pos = np.array([200, 200, 200])
#         objA.set_position_orientation(old_pos, orientation)

#         if sample_on_floor:
#             _, pos = objB.scene.get_random_point_by_room_instance(objB.room_instance)

#             if pos is not None:
#                 # Get the combined AABB.
#                 lower, _ = objA.states[object_states.AABB].get_value()
#                 # Move the position to a stable Z for the object.
#                 pos[2] += objA.get_position()[2] - lower[2]
#         else:
#             if use_ray_casting_method:
#                 if predicate == "onTop":
#                     params = _ON_TOP_RAY_CASTING_SAMPLING_PARAMS
#                 elif predicate == "inside":
#                     params = _INSIDE_RAY_CASTING_SAMPLING_PARAMS
#                 else:
#                     assert False, "predicate is not onTop or inside: {}".format(predicate)

#                 # Retrieve base CoM frame-aligned bounding box parallel to the XY plane
#                 parallel_bbox_center, parallel_bbox_orn, parallel_bbox_extents, _ = objA.get_base_aligned_bounding_box(
#                     xy_aligned=True
#                 )

#                 # TODO: Get this to work with non-URDFObject objects.
#                 sampling_results = sampling_utils.sample_cuboid_on_object(
#                     objB,
#                     num_samples=1,
#                     cuboid_dimensions=parallel_bbox_extents,
#                     axis_probabilities=[0, 0, 1],
#                     refuse_downwards=True,
#                     undo_padding=True,
#                     **params,
#                 )

#                 sampled_vector = sampling_results[0][0]
#                 sampled_quaternion = sampling_results[0][2]

#                 sampling_success = sampled_vector is not None
#                 if sampling_success:
#                     # Move the object from the original parallel bbox to the sampled bbox
#                     parallel_bbox_rotation = R.from_quat(parallel_bbox_orn)
#                     sample_rotation = R.from_quat(sampled_quaternion)
#                     original_rotation = R.from_quat(orientation)

#                     # The additional orientation to be applied should be the delta orientation
#                     # between the parallel bbox orientation and the sample orientation
#                     additional_rotation = sample_rotation * parallel_bbox_rotation.inv()
#                     combined_rotation = additional_rotation * original_rotation
#                     orientation = combined_rotation.as_quat()

#                     # The delta vector between the base CoM frame and the parallel bbox center needs to be rotated
#                     # by the same additional orientation
#                     diff = old_pos - parallel_bbox_center
#                     rotated_diff = additional_rotation.apply(diff)
#                     pos = sampled_vector + rotated_diff
#             else:
#                 random_idx = np.random.randint(len(objB.supporting_surfaces[predicate].keys()))
#                 body_id, link_id = list(objB.supporting_surfaces[predicate].keys())[random_idx]
#                 random_height_idx = np.random.randint(len(objB.supporting_surfaces[predicate][(body_id, link_id)]))
#                 height, height_map = objB.supporting_surfaces[predicate][(body_id, link_id)][random_height_idx]
#                 obj_half_size = np.max(objA.bounding_box) / 2 * 100
#                 obj_half_size_scaled = np.array([obj_half_size / objB.scale[1], obj_half_size / objB.scale[0]])
#                 obj_half_size_scaled = np.ceil(obj_half_size_scaled).astype(int)
#                 height_map_eroded = cv2.erode(height_map, np.ones(obj_half_size_scaled, np.uint8))

#                 valid_pos = np.array(height_map_eroded.nonzero())
#                 if valid_pos.shape[1] != 0:
#                     random_pos_idx = np.random.randint(valid_pos.shape[1])
#                     random_pos = valid_pos[:, random_pos_idx]
#                     y_map, x_map = random_pos
#                     y = y_map / 100.0 - 2
#                     x = x_map / 100.0 - 2
#                     z = height

#                     pos = np.array([x, y, z])
#                     pos *= objB.scale

#                     # the supporting surface is defined w.r.t to the link frame, not
#                     # the inertial frame
#                     if link_id == -1:
#                         link_pos, link_orn = pb.getBasePositionAndOrientation(body_id)
#                         dynamics_info = pb.getDynamicsInfo(body_id, -1)
#                         inertial_pos = dynamics_info[3]
#                         inertial_orn = dynamics_info[4]
#                         inv_inertial_pos, inv_inertial_orn = pb.invertTransform(inertial_pos, inertial_orn)
#                         link_pos, link_orn = pb.multiplyTransforms(
#                             link_pos, link_orn, inv_inertial_pos, inv_inertial_orn
#                         )
#                     else:
#                         link_pos, link_orn = get_link_pose(body_id, link_id)
#                     pos = matrix_from_quat(link_orn).dot(pos) + np.array(link_pos)
#                     # Get the combined AABB.
#                     lower, _ = objA.states[object_states.AABB].get_value()
#                     # Move the position to a stable Z for the object.
#                     pos[2] += objA.get_position()[2] - lower[2]

#         if pos is None:
#             success = False
#         else:
#             pos[2] += z_offset
#             objA.set_position_orientation(pos, orientation)
#             success = not any(detect_closeness(bid) for bid in objA.get_body_ids())

#         if igibson.debug_sampling:
#             print("sample_kinematics", success)
#             embed()



#         if success:
#             break
#         else:
#             restoreState(state_id)

#     pb.removeState(state_id)

#     if success:
#         if not skip_falling:
#             objA.set_position_orientation(pos, orientation)

#             # Let it fall for 0.2 second
#             physics_timestep = pb.getPhysicsEngineParameters()["fixedTimeStep"]
#             for _ in range(int(0.2 / physics_timestep)):
#                 pb.stepSimulation()
#                 if any(detect_collision_with_others(bid) for bid in objA.get_body_ids()):
#                     break

#         pose = objA.get_position_orientation()
#     else:
#         pose = None
    
#     restoreState(state_id)
#     return pose


