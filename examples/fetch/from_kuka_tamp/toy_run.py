


from collections import UserDict
from itertools import takewhile
from json import load
import os
import functools as ft
import inspect as ins
import sys
import numpy as np
from typing import Any, List, NamedTuple, NamedTuple, NoReturn, Optional, Tuple, Iterable, Union
from igibson.tasks.behavior_task import is_mesh_on_surface, is_movable

from igibson.utils.assets_utils import get_all_object_categories, get_ig_model_path, get_object_models_of_category
from numpy import random
from transforms3d.euler import quat2euler
from examples.pybullet.utils.pybullet_tools.utils import Attachment
from iGibson.igibson.utils.assets_utils import get_ig_avg_category_specs

from igibson.action_primitives.starter_semantic_action_primitives import URDFObject
from igibson.objects.object_base import BaseObject

from examples.fetch.from_kuka_tamp.fetch_primitives import BID, UID, MyiGibsonSemanticInterface
from examples.fetch.from_kuka_tamp.utils.object_spec import Euler, Kwargs, ObjectSpec, Orientation3D, Position3D, Quaternion
from examples.pybullet.utils.pybullet_tools.kuka_primitives import Attach, BodyConf, BodyGrasp, BodyPath, BodyPose
from pddlstream.algorithms.meta import create_parser, solve
from pddlstream.language.constants import PDDLProblem
from pddlstream.language.generator import from_fn, from_gen_fn, from_test
from pddlstream.utils import Profiler, read

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
        self.data.update(dict(
            zip(('filename', 'model_path'), self.extract_file_info(category, **kwargs))
        ))
        self.data.update(kwargs)
        
        self._state_keys = [key for key in self.data if key not in ObjSpec.URDF_ARGS]
        self._urdf_keys  = [key for key in self.data if key in ObjSpec.URDF_ARGS]

    def extract_file_info(self, category, *, model=None, model_path=None, **kwargs):
        if category not in get_all_object_categories():
            raise ValueError(f"Unable to find object category '{category}' in assets.")
        if model is None: 
            model = random.choice(get_object_models_of_category(category))
        if model_path is None:
            model_path = get_ig_model_path(self.data["category"], model)
        filename = os.path.join(model_path, model + ".urdf")
        return filename, model_path

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
            
                

            
        


def load_object(sim:MyiGibsonSemanticInterface, spec:ObjSpec, verbose:bool=True):
    URDF_kwargs = {
            'avg_obj_dims' : get_ig_avg_category_specs().get(spec.category),
            'fit_avg_dim_volume' : True,
            'texture_randomization' : False,
            'overwrite_inertial' : True,
        }
        
    URDF_kwargs.update(**spec.urdf_kwargs)

    if verbose: 
        print(
            f"Loading {spec.category.capitalize()} object '{spec.name}'...",
            end=''
        )  
    
    # Create and import the object
    sim_obj = URDFObject(**URDF_kwargs)
    sim.env.simulator.import_object(sim_obj)

    if verbose: 
        print(" done.")
            
    # Return obj body_ids
    return sim.get_bid(sim_obj)

def init_object_state(  
                sim:MyiGibsonSemanticInterface,
                name:str,
                position:Position3D=(0,0,0), 
                orientation:Orientation3D=Euler(0,0,0), 
                on_floor:bool=False, 
                **state:Kwargs
):
    if len(orientation)==4:
        orientation = quat2euler(orientation)
    if on_floor:
        sim.env.land(sim.get_object(name), position, orientation)
    else:
        sim.set_position_orientation(name, position, orientation)
    
    if not state=={}:
        raise NotImplementedError(f"No handling defined for state variables {list(state)}")
        
class ModifiedMiGSI(MyiGibsonSemanticInterface):

    def __init__(self, config_file:str, objects:Iterable[ObjSpec]=[], *, headless:bool=True, verbose=True):
        self._init_igibson(config=config_file, headless=headless)
        self._load_objects(objects)
        self._init_objects(objects)
        self._init_robot_state()
    
    def _load_objects(self, specs:Iterable[ObjSpec], verbose: bool = True) -> List[BID]:
        def load_single_object(spec:ObjSpec):
            if verbose: print(f"Loading {spec.category.capitalize()} object '{spec.name}'...", end='')  

            URDF_kwargs = {
                'avg_obj_dims' : get_ig_avg_category_specs().get(spec.category),
                'fit_avg_dim_volume' : True,
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
                      on_floor:bool=False, 
                      **state:Kwargs) -> None:
        if len(orientation)==4:
            orientation = quat2euler(orientation)
        if on_floor:
            self.env.land(self.get_object(name), position, orientation)
        else:
            self.set_position_orientation(name, position, orientation)
        if not state=={}:
            raise NotImplementedError(f"No handling defined for state variables {list(state)}")
    
    def _init_objects(self, specs:Iterable[ObjSpec]) -> None:
        for spec in specs:
            self._init_object_statevars(spec.name, **spec.state)
    
    def _init_object_state(self, obj_specs: Iterable[ObjectSpec], *, verbose=True) -> None:
        raise NotImplementedError


    def is_surface(self, obj:UID):
        return not self.is_movable(obj) and (self.get_object(obj).category.lower() not in ['walls', 'ceilings'])
    
    # def sample_placement_BodyPose(self, obj:UID, surface:UID) -> BodyPose:
    #     placement = self.sample_placement(obj, surface)
    #     return BodyPose(obj, placement)




def init_sim(objects:Iterable[ObjSpec]=[]):
    parser = create_parser()
    parser.add_argument('-g','--gui', action='store_true', help='Render and visualize the system')
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = "fetch_tamp.yaml"
    config_path = os.path.join(dir_path,config)

    sim = ModifiedMiGSI(
        config_file=config_path, 
        objects=objects,
        headless=(not args.gui)
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
    
    g = next(sim.get_grasp_gen()('celery'))[0]
    init = [
        ('AtConf', q_init),
        # ('Grasp', g),
        # ('GraspForObj', 'celery', g),
        # ('Grasping', 'celery', g)
    ]
    for sbj in _subjects:
        if isinstance(sbj,BodyConf):
            init += [('Conf', sbj)]
        elif sbj in sim.objects:            
            ObjType = sim.get_object(sbj).category.capitalize()
            init += [('Obj', sbj)] #, (ObjType, sbj)]
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
    goal = ('On', 'celery', 'stove')
    # goal = ( 'exists', ('?p'), ('Placement', '?p', 'celery', 'stove'))

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



def main():
    DEBUG = True

    sink   = ObjSpec("sink",  "sink",    position=(-0.5,  0.0, 0.0), on_floor=True, fixed_base=True)
    stove  = ObjSpec("stove",  "stove",  position=(+0.5,  0.0, 0.0), on_floor=True, fixed_base=True)
    celery = ObjSpec("celery", "celery", position=( 0.0, +0.5, 0.0), on_floor=True, fixed_base=False)
    radish = ObjSpec("radish", "radish", position=( 0.0, -0.5, 0.0), on_floor=True, fixed_base=False)

    sim = init_sim(objects=[sink,stove,celery,radish])
    
    qinit = sim.get_arm_config()
    if DEBUG: 
        qgoal = (0.12424097874999764, 1.3712678552985822, -0.3116053947850258, 2.62593026717931, 1.190838900298941, 0.2317687545849476, 1.0115493242626759, -1.2075981800730915) # avoid spending time sampling during debugging
    else:
        qgoal = sim.sample_arm_config(max_attempts=100)
    print(qinit)
    print(qgoal)

    plan = motion_plan(sim, qinit, qgoal)
    print('motion plan length:', len(plan))
    print('\n\n\n')

    problem = get_pddlproblem(sim, qinit, qgoal)
    solution = solve(problem, unit_costs=True, debug=DEBUG)
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







if __name__=='__main__':
    main()