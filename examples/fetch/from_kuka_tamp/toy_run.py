


from collections import UserDict
from itertools import takewhile
from json import load
import os
import functools as ft
import inspect as ins
from time import sleep
import cv2
import sys
from typing_extensions import TypeAlias
from igibson import object_states
from igibson.external.pybullet_tools.utils import refine_path, wait_for_user
from igibson.robots import Fetch
import numpy as np
from typing import Any, List, NamedTuple, NamedTuple, NewType, NoReturn, Optional, Tuple, Iterable, Union
from igibson.tasks.behavior_task import enable_real_time, is_mesh_on_surface, is_movable

from igibson.utils.assets_utils import get_all_object_categories, get_ig_model_path, get_object_models_of_category
from numpy import pi, random
from transforms3d.euler import quat2euler
from examples.pybullet.utils.pybullet_tools.utils import Attachment, trajectory_controller, wait_for_duration
from iGibson.igibson.utils.assets_utils import get_ig_avg_category_specs

from igibson.action_primitives.starter_semantic_action_primitives import URDFObject
from igibson.objects.object_base import BaseObject

from examples.fetch.from_kuka_tamp.fetch_primitives import BID, UID, JointPos, MyiGibsonSemanticInterface, Object
from examples.fetch.from_kuka_tamp.utils.object_spec import Euler, Kwargs, ObjectSpec, Orientation3D, Position3D, Quaternion
from examples.pybullet.utils.pybullet_tools.kuka_primitives import Attach, BodyConf, BodyGrasp, BodyPose, BodyPath, Command
from pddlstream.algorithms.meta import create_parser, solve
from pddlstream.language.constants import PDDLProblem
from pddlstream.language.generator import from_fn, from_gen_fn, from_test
from pddlstream.utils import Profiler, read

##############################################################################

Pose: TypeAlias = Tuple[Position3D, Orientation3D]



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

    def __init__(self, config_file:str, objects:Iterable[ObjSpec]=[], headless:bool=True, 
                 *, robot_pose:Pose=((0,0,0),(0,0,0)), viewer_pose:Optional[Pose]=None, verbose=True, **config_options:Kwargs):
        self._init_igibson(config=config_file, headless=headless, **config_options)
        # self._robot.controller_config
        self._load_objects(objects)
        self._init_objects(objects)
        self._init_robot_state(*robot_pose)
        if not headless:
            self._viewer_setup(*viewer_pose) if viewer_pose is not None else self._viewer_setup()

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
            self.land(obj, position, orientation)
            # print(position)

            print(obj.states[object_states.OnTop].get_value(surface)) #, use_ray_casting_method=True)
        else:
            self.set_position_orientation(name, position, orientation)
        if not state=={}:
            raise NotImplementedError(f"No handling defined for state variables {list(state)}")
    
    def _init_objects(self, specs:Iterable[ObjSpec]) -> None:
        pose = self.get_pose(self.robot)
        self.set_position(self.robot, (100,100,100))
        for spec in specs:
            self._init_object_statevars(spec.name, **spec.state)
        self.set_pose(self.robot, pose)
    
    def _init_object_state(self, obj_specs: Iterable[ObjectSpec], *, verbose=True) -> None:
        raise NotImplementedError


    def is_surface(self, obj:UID):
        return not self.is_movable(obj) and (self.get_object(obj).category.lower() not in ['walls', 'ceilings'])
    
    def land(self, obj:Object, position:Optional[Position3D]=None, orientation:Optional[Orientation3D]=None):
        if position is None:    position = self.get_position(obj)
        if orientation is None: orientation = self.get_orientation(obj)
        if len(orientation)==4: orientation = quat2euler(orientation)

        obj = self.get_object(obj)
        if obj.fixed_base:
            print(f"WARNING: {self.__class__.__name__}.land() called on object '{obj.name}' with fixed_base=True.")
        self.env.land(obj, position, orientation)





class ArmCommand(Command):

    def control(self, real_time=False, dt=0.05, verbose=False):
        for body_path in self.body_paths:
            body_path.control(real_time=real_time, dt=dt) #, verbose=verbose)

    
    def execute(self, time_step=0.05):
        sim = self.body_paths[0].sim
        for i, body_path in enumerate(self.body_paths):
            for q in body_path.iterator():
                wait_for_duration(time_step)

class ArmPath(BodyPath):
    def __init__(self, sim:ModifiedMiGSI, path:Iterable[JointPos], attachments:List[Attachment]=[]):
        super().__init__(sim.robot_bid, path, sim._arm_joint_ids, attachments)
        self.sim = sim

    # def executor(self): # more accurate name
    def iterator(self):
        for configuration in self.path:
            self.sim.set_arm_config(configuration, self.attachments)
            self.sim.env.simulator.sync()
            yield configuration

    def refine(self, num_steps=0, update=False):
        refined_path = self.__class__(self.sim, refine_path(self.body, self.joints, self.path, num_steps), self.attachments)
        if update:
            self.path = refined_path
        return refined_path
    
    # def control(self, real_time=False, dt=0.05, verbose=False):
    #     for configuration in self.path:
            
    #         for q in trajectory_controller(self.body, self.joints, self.path):
    #             self.sim.env.simulator_step()

            # action = np.zeros(self.sim.env.action_space.shape)
            # action[4:12] = configuration

            # q = self.sim.get_arm_config()
            # while not np.allclose(q, configuration, atol=1e-3, rtol=0):
            #     state, reward, done, x = self.sim.env.step(action)
            #     q = self.sim.get_arm_config()
            # self.sim.env.simulator.step()
            # wait_for_duration(dt)

            # if verbose:
            #     print('Action:', [x for x in action])
            #     print('Config:', self.sim.get_arm_config())
            #     print()
    




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


    goal = ('AtConf', q_goal)
    # goal = ('Holding', 'celery')
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
        'motion-plan' : from_fn(motion_planner),
        'motion-plan-carry' : from_fn(carry_planner),
    }


    problem=PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)
    return problem


def run_planner(sim:ModifiedMiGSI, qgoal:Optional[Union[float,BodyConf]]=None, debug:bool=False, print_results:bool=True):
    qinit = sim.get_arm_config()
    if qgoal is None: 
        # qgoal = np.array(qinit) + np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        qgoal = sim._robot.tucked_default_joint_pos[sim._arm_joint_control_indices]
        # qgoal =  (0.385, -1.06595, -0.22184, 1.53448, 1.46076, -0.84995, 1.36904, 1.90996)
        # qgoal = (0.237, -0.957, -0.212, -2.493, 1.488, -0.736, -1.562, -2.875)
        # qgoal = (0.12424097874999764, 1.3712678552985822, -0.3116053947850258, 2.62593026717931, 1.190838900298941, 0.2317687545849476, 1.0115493242626759, -1.2075981800730915) 

    problem = get_pddlproblem(sim, qinit, qgoal)
    solution = solve(problem, unit_costs=True, debug=debug)
    if print_results:
        print(solution)
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
                    #   position=(-2.36734748,  5.5552516 ,  0.45443946),
                      orientation=(0,0,pi/2),
                    #   orientation=(-0.        , -0.        ,  0.70853085,  0.70567984), 
                      fixed_base=True,
                    #   scale=np.array((0.5, 0.5, 0.7)),
                    #   scale=np.array((0.44914447, 0.52874679, 0.63933571)),
                    #   bounding_box=(0.496562, 0.614757, 0.891),
                      bounding_box=(1.0, 0.5, 0.75),
                      )
    sink   = ObjSpec("sink",  "sink", model='kitchen_sink',    
                     position=(-2.35,  5.8,  0.3),
                     orientation=(0,0,pi/2),
                     fixed_base=True,
                    #  scale=np.array([0.97625266, 0.70599113, 0.78163018]),
                     bounding_box=(0.75,  0.5,  0.75),
                    #  bounding_box=(0.88,  0.62,  1.089),
                     )
    stove  = ObjSpec("stove",  "stove", model='101924',
                     position=(-0.225  , 5.75,  0.6), 
                    #  position=(-0.221481  ,  5.92155266,  0.52469647), 
                     orientation=(0,0,-pi/2),
                     fixed_base=True,
                    #  scale=np.array((0.68030202, 0.93826808, 0.78960697)), 
                     bounding_box=(1.0,  0.75,  1.25),
                    )
    celery = ObjSpec("celery", "celery", model='celery_000', 
                     position=(-2.3,  5.1,  1.0), place_on='counter', fixed_base=False, scale=0.15)
    radish = ObjSpec("radish", "radish", model='45_0', 
                     position=(-2.15,  5.3,  1.0), place_on='counter', fixed_base=False, scale=1.25)
    
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


def main():
    parser = create_parser()
    parser.add_argument('-g','--gui', action='store_true', help='Render and visualize the system')
    args = parser.parse_args()

    DEBUG = True
    SIMULATE = True
    GUI = args.gui if (gui_env_var:=os.getenv("GUI")) is None else gui_env_var
    if GUI:
        print("Visualization on.")

    sim = init_sim(objects=[], # get_objects(), 
                   headless=(not GUI), 
                   visualize_planning=True)
    try:
        qinit = sim._robot.tucked_default_joint_pos[sim._arm_joint_control_indices]
        qgoal = sim._robot.untucked_default_joint_pos[sim._arm_joint_control_indices]
        # qgoal = (0.12424097874999764, 1.3712678552985822, -0.3116053947850258, 2.62593026717931, 1.190838900298941, 0.2317687545849476, 1.0115493242626759, -1.2075981800730915) 

        sim.set_arm_config(qinit)
        sim.env.simulator.viewer.initial_pos = [-1.9, 5.1, 1,1]
        sim.env.simulator.viewer.initial_view_direction = [0.6, 0.6, -0.4]
        sim.env.simulator.viewer.reset_viewer()
        sim.env.simulator.sync()


        solution = run_planner(sim, 
                               qgoal=qgoal,
                            #    debug=DEBUG, 
                               print_results=True
                            )
        plan = solution.plan

        # if GUI:
        #     visualize_sim(sim)


        

        if (plan is not None) and GUI:
            command = consolidate_plan(plan)

            path = plan[0].args[-1].path
            [print(q) for q in path[-5:]]
            print()
            
            if SIMULATE:
                wait_for_user('Simulate?')
                command.control()
            else:
                wait_for_user('Execute?')
                #command.step()
                # command.refine(num_steps=10).execute(time_step=0.001)
                command.refine(num_steps=10).execute(time_step=0.001)
            print("Config: ", sim.get_arm_config())
            wait_for_user('Show qgoal?')
            sim.set_arm_config(qgoal, freeze=True)
            print("Config: ", sim.get_arm_config())
            sim.env.simulator.step()
            print()
            print(path[0])
            print(path[-1])
            print()
            print("Press 'esc' on Viewer window to exit.")
            while True:
                sim.env.simulator.step()

    finally:
        sim.env.close()

    
    

    







if __name__=='__main__':
    main()