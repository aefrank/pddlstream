


from collections import UserDict
from itertools import takewhile
from json import load
import os
import functools as ft
import inspect as ins
import sys
from typing_extensions import TypeAlias
from igibson import object_states
import numpy as np
from typing import Any, List, NamedTuple, NamedTuple, NewType, NoReturn, Optional, Tuple, Iterable, Union
from igibson.tasks.behavior_task import is_mesh_on_surface, is_movable

from igibson.utils.assets_utils import get_all_object_categories, get_ig_model_path, get_object_models_of_category
from numpy import pi, random
from transforms3d.euler import quat2euler
from examples.pybullet.utils.pybullet_tools.utils import Attachment
from iGibson.igibson.utils.assets_utils import get_ig_avg_category_specs

from igibson.action_primitives.starter_semantic_action_primitives import URDFObject
from igibson.objects.object_base import BaseObject

from examples.fetch.from_kuka_tamp.fetch_primitives import BID, UID, MyiGibsonSemanticInterface, Object
from examples.fetch.from_kuka_tamp.utils.object_spec import Euler, Kwargs, ObjectSpec, Orientation3D, Position3D, Quaternion
from examples.pybullet.utils.pybullet_tools.kuka_primitives import Attach, BodyConf, BodyGrasp, BodyPath, BodyPose
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
                      land:bool=False, 
                      **state:Kwargs) -> None:
        if land:
            self.land(name, position, orientation)
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




def init_sim(objects, headless=True, **kwargs):
    
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
def motion_plan(sim:ModifiedMiGSI, q1, q2, o=None, g=None):
    if isinstance(q1,BodyConf): q1 = q1.configuration
    if isinstance(q2,BodyConf): q2 = q2.configuration
    if o is not None:
        assert g is not None and g.body==sim.get_name(o)
        attachments = [g.attachment()]
    else:
        attachments=[]
    return (BodyPath(sim.robot_bid, sim.plan_arm_joint_motion(q1,q2,attachments=attachments)),)

def plan_grasp_and_carry(sim:ModifiedMiGSI, q1, q2, obj, g):
    if isinstance(q1,BodyConf): q1 = q1.configuration
    if isinstance(q2,BodyConf): q2 = q2.configuration
    # grasp = next(sim.get_grasp_gen()(obj))[0]
    attachment = g.attachment()
    path = BodyPath(sim.robot_bid, sim.plan_arm_joint_motion(q1,q2,attachments=[attachment]))
    return(path,)



def get_pddlproblem(sim:ModifiedMiGSI, q_goal:Union[float,BodyConf], q_init:Optional[Union[float,BodyConf]]=None):
    if q_init is None:                     
        q_init = BodyConf(sim.robot_bid, sim.get_arm_config())
    else:
        sim.set_arm_config(q_init)
        if not isinstance(q_init, BodyConf): 
            q_init = BodyConf(sim.robot_bid, q_init)
    if not isinstance(q_goal, BodyConf): 
        q_goal = BodyConf(sim.robot_bid, q_goal)

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
    # goal = ('Holding', 'celery')
    # goal = ('On', 'celery', 'stove')
    goal = ('Cooked', 'radish')

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
    stream_pddl =  read(os.path.join(os.path.dirname(os.path.
    realpath(__file__)), 'toy_stream.pddl'))
    
    sample_arm_config = sim.sample_arm_config
    motion_planner = ft.partial(motion_plan, sim)
    carry_planner = ft.partial(plan_grasp_and_carry, sim)
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


def run_planner(sim:ModifiedMiGSI, debug:bool=False, display:bool=True):
    qinit = sim.get_arm_config()
    if debug: 
        qgoal = (0.12424097874999764, 1.3712678552985822, -0.3116053947850258, 2.62593026717931, 1.190838900298941, 0.2317687545849476, 1.0115493242626759, -1.2075981800730915) # avoid spending time sampling during debugging
    else:
        qgoal = sim.sample_arm_config(max_attempts=100)
    print(qinit)
    print(qgoal)

    plan = motion_plan(sim, qinit, qgoal)
    print('motion plan length:', len(plan))
    print('\n\n\n')

    problem = get_pddlproblem(sim, qinit, qgoal)
    solution = solve(problem, unit_costs=True, debug=debug)
    if display:
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

def visualize_sim(sim:ModifiedMiGSI):
    sim.env.simulator.viewer.reset_viewer()

    while True:
        action = np.zeros(sim.env.action_space.shape)
        state, reward, done, x = sim.env.step(action)
        # print(state, reward, done, x)
        # print(sim.get_pose('stove'))
        # print(sim.get_object('stove').states[object_states.AABB].get_value())


def main():
    parser = create_parser()
    parser.add_argument('-g','--gui', action='store_true', help='Render and visualize the system')
    args = parser.parse_args()

    DEBUG = True
    GUI = args.gui if (gui_env_var:=os.getenv("USE_GUI")) is None else gui_env_var

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
                     position=(-2.3,  5.1,  1.0), land=True, fixed_base=False, scale=0.15)
    radish = ObjSpec("radish", "radish", model='45_0', 
                     position=(-2.15,  5.3,  1.0), land=True, fixed_base=False, scale=1.25)
    objects = [sink, stove, counter, celery, radish]

    sim = init_sim(objects=objects, headless=(not GUI), visualize_planning=True)

    # for attr in dir(sink_obj:=sim.get_object('sink')):
    #     print(attr, getattr(sink_obj,attr))
    for obj in sim._objects:
        print(obj.name, obj.scale, obj.bounding_box, obj.get_position_orientation())
    visualize_sim(sim)

    # solution = run_planner(sim, debug=DEBUG, display=True)

    sim.env.close()

    
    

    







if __name__=='__main__':
    main()