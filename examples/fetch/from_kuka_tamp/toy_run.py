


from collections import UserDict
from itertools import takewhile
from json import load
import os
import functools as ft
import inspect as ins
import sys
from typing import Any, NamedTuple, NamedTuple, Optional, Tuple, Iterable

from igibson.utils.assets_utils import get_all_object_categories, get_ig_model_path, get_object_models_of_category
from numpy import random
from transforms3d.euler import quat2euler
from iGibson.igibson.utils.assets_utils import get_ig_avg_category_specs

from igibson.action_primitives.starter_semantic_action_primitives import URDFObject
from igibson.objects.object_base import BaseObject

from examples.fetch.from_kuka_tamp.fetch_primitives import UID, MyiGibsonSemanticInterface
from examples.fetch.from_kuka_tamp.utils.object_spec import Euler, Kwargs, Orientation3D, Position3D, Quaternion
from examples.pybullet.utils.pybullet_tools.kuka_primitives import BodyConf, BodyPath
from pddlstream.algorithms.meta import create_parser, solve
from pddlstream.language.constants import PDDLProblem
from pddlstream.language.generator import from_fn
from pddlstream.utils import Profiler, read

##############################################################################







def motion_plan(sim, q1, q2):
    if isinstance(q1,BodyConf): q1 = q1.configuration
    if isinstance(q2,BodyConf): q2 = q2.configuration
    return (BodyPath(sim.robot_bid, sim.plan_arm_joint_motion(q1,q2)),)


def get_pddlproblem(sim, q1, q2):
    if not isinstance(q1,BodyConf): q1 = BodyConf(sim.robot_bid, q1)
    if not isinstance(q2,BodyConf): q2 = BodyConf(sim.robot_bid, q2)

    domain_pddl = read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'toy_domain.pddl'))
    constant_map = {}
    stream_pddl =  read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'toy_stream.pddl'))
    
    motion_planner = ft.partial(motion_plan, sim)
    sample_arm_config = sim.sample_arm_config
    stream_map = {
        'sample-conf' : from_fn(sample_arm_config),
        'motion-plan' : from_fn(motion_planner)
    }

    init = [("Conf", q1), ("AtConf", q1), ("Conf", q2)]
    goal = ("AtConf", q2)
    
    problem=PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)
    return problem

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
        


        
        # try:
        #     return object.__getattribute__(self, attr)
        # except AttributeError as e:
        #     if attr!='data': # avoid loop
        #         try:
        #             data = object.__getattribute__(self,'data')
        #             if attr in data:
        #                 return data[attr]
        #         except Exception:
        #             pass
        #         raise e

        # if attr!='data' and not self.__hasattr__(attr) and self.__hasattr__('data'):
        #     data = object.__getattribute__(self, 'data')
        #     if attr in data:
        #         return data[attr]
        # return object.__getattribute__(self,attr)
        

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
            
                
                

        # try:
        #     object.__getattribute__(self, attr) # check if self.attr exists
        # except AttributeError as e:
        #     if attr!='data':
        #         try:
        #             data = object.__getattribute__(self,'data')
        #             if attr in data:
        #                 data[attr] = value
        #                 return
        #         except Exception:
        #             pass
        # object.__setattr__(self, attr, value)
            
        


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

def init_object_state(  sim:MyiGibsonSemanticInterface,
                        name:str,
                        position:Position3D=(0,0,0), 
                        orientation:Orientation3D=Euler(0,0,0), 
                        on_floor:bool=False, 
                        **state:Kwargs):
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
        for obj in objects:
            load_object(self, obj)
        for obj in objects:
            init_object_state(self, obj.name, **obj.state)
        self._init_robot_state()

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

def main():
    sink   = ObjSpec("sink",  "sink",    position=(-0.5,  0.0, 0.0), on_floor=True, fixed_base=True)
    stove  = ObjSpec("stove",  "stove",  position=(+0.5,  0.0, 0.0), on_floor=True, fixed_base=True)
    celery = ObjSpec("celery", "celery", position=( 0.0, +0.5, 0.0), on_floor=True, fixed_base=False)
    radish = ObjSpec("radish", "radish", position=( 0.0, -0.5, 0.0), on_floor=True, fixed_base=False)
    # print(celery)
    # print(celery.urdf_kwargs)
    # print(celery.properties)
    celery.name = "cel"

    sim = init_sim(objects=[sink,stove,celery,radish])
    print(sim.objects)
    print(celery)
    # for obj in (sink,stove,celery,radish):
    #     load_object(sim, obj)
    # for obj in (sink,stove,celery,radish):
    #     init_object_state(sim, obj['name'], **obj.state)
    
    

    # qinit = sim.get_arm_config()
    # qgoal = sim.sample_arm_config(max_attempts=100)
    # print(qinit)
    # print(qgoal)

    # plan = motion_plan(sim, qinit, qgoal)
    # print('motion plan length:', len(plan))
    # print('\n\n\n')

    # problem = get_pddlproblem(sim, qinit, qgoal)
    # solution = solve(problem, unit_costs=True, debug=True)
    # print(solution)







if __name__=='__main__':
    main()