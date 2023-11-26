
import itertools
import os
from typing import Optional, Union
import numpy as np

from igibson.robots import Fetch
from examples.pybullet.utils.pybullet_tools.kuka_primitives import BodyConf, BodyPose
from iGibson.igibson.termination_conditions import predicate_goal
from pddlstream.algorithms.downward import parse_problem, parse_sequential_domain
from pddlstream.language.constants import PDDLProblem
from pddlstream.utils import read


class ArmConf(BodyConf):
    n = itertools.count()

    def __init__(self, sim, configuration):
        if np.isscalar(configuration):
            configuration = [configuration]
        # super().__init__(body=sim.robot_bid, joints=sim.arm_joint_idx, configuration=configuration)
        super().__init__(sim.robot_bid, joints=range(len(configuration)), configuration=configuration)
        self.sim = sim

class MySimInterface():

    def __init__(self, domain_filename='domain.pddl'):
        if not os.path.isabs(domain_filename): domain_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), domain_filename) 
        self.domain = parse_sequential_domain(read(domain_filename))
        self.state = {predicate.name:({} if predicate.arguments else None) for predicate in self.domain.predicates}
        self.state["fetch"] = "fetch"



    def set_state_variables(self, *predicates, assume_binary_relations=False):
        if assume_binary_relations:
            for name, *args in predicates:
                property_dict = self.state[name.lower()]
                if args[0] not in property_dict:
                    property_dict[args[0]] = set()
                property_dict[args[0]].add(tuple(args[1:]))
        else:
            for name, *args in predicates:
                # if args:
                #     self.state.update({name:{args[0]:{tuple(args[1:]):True}}})
                # else:
                    self.state[name.lower()].add(tuple(args))


    
    @property
    def robot(self):
        return self.state["fetch"]
    
    @property
    def robot_bid(self):
        return 0
    
    def get_arm_config(self):
        if not "ArmConf" in self.state: 
            self.set_arm_config((0,0,0))
        return self.state["ArmConf"]
    def set_arm_config(self, q):
        if not isinstance(q, ArmConf):
            q = ArmConf(self, q)
        self.state["ArmConf"] = q

    def get_pose(self, obj):
        return self.state["AtPose"][obj]
    def set_pose(self, obj, value):
        self.state["AtPose"][obj] = value

    def is_movable(self, obj, value=None):
        if value is not None:
            assert value in [True, False]
            self.state["Movable"][obj] = value
        return self.state["Movable"][obj]
    
    def is_on(self, obj, surface, value=None):
        if value is not None:
            assert value in [True, False]
            self.state["On"][(obj,surface)] = value
        return self.state["On"][(obj,surface)]
    

    @property
    def PREDICATE_NAMES_MAP(self):
        {
            'arm_config' : "ArmConf",
            
        }

    def get_pddl_state(self):
        for key, values in self.data.items():
            pass

        
    
    


# def get_pddlproblem(sim:MySimInterface, q_init, q_goal, verbose=True):
#     if q_init is not None:                     
#         sim.set_arm_config(q_init)
#     q_init = sim.get_arm_config()
#     if not isinstance(q_goal, ArmConf): 
#         q_goal = ArmConf(sim, q_goal)

#     if verbose:
#         print()
#         print('Initial configuration: ', q_init.configuration)
#         print('Goal configuration: ', q_goal.configuration)
#         print()

#     # objects  = [sim.get_bid(obj) for obj in sim.objects]

#     _subjects = [q_init, q_goal, *sim.objects]
    
#     init = [
#         ('AtArmConf', q_init),
#     ]
#     for sbj in _subjects:
#         if isinstance(sbj, ArmConf):
#             init += [('ArmConf', sbj)]
#         elif sbj in sim.objects:            
#             init += [('Obj', sbj)]
#             if sim.is_movable(sbj):
#                 pose = BodyPose(sbj, [tuple(x) for x in sim.get_pose(sbj)]) # cast back from np.array for less visual clutter
#                 init +=[
#                     ('Pose', pose),
#                     ('AtPose', sbj, pose),
#                     ('Movable', sbj)
#                 ]
#                 # find surface that it is on
#                 for _obj in sim.objects:
#                     if not sim.is_movable(_obj) and not _obj in ['walls', 'ceilings']:
#                         if sim.is_on(sbj, _obj):
#                             init += [('Placement', pose, sbj, _obj)]


#     # goal = ('AtConf', q_goal)
#     goal = ('Holding', 'radish')
#     # goal = ('On', 'celery', 'stove')
#     # goal = ('Cooked', 'radish')

#     if verbose:
#         print(f"\n\nINITIAL STATE")
#         for fluent in init:
#             fluent = tuple(
#                 (sim.get_name(arg) if isinstance(arg,int) else 
#                 tuple(np.round(arg.configuration,3)) if isinstance(arg,BodyConf) else 
#                 (sim.get_name(arg.body), *(tuple(np.round(p,3)) for p in arg.pose)) if isinstance(arg, BodyPose) else 
#                 arg) for arg in fluent
#             )

#             print(fluent)
#         print('\n')


#     domain_pddl = read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'toy_domain.pddl'))
#     constant_map = {}
#     stream_pddl =  read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'toy_stream.pddl'))
    
#     sample_arm_config = sim.sample_arm_config
#     motion_planner = ft.partial(motion_plan, sim, as_tuple=True)
#     carry_planner = ft.partial(plan_grasp_and_carry, sim, as_tuple=True)
#     stream_map = {
#         'sample-conf' : from_fn(sample_arm_config),
#         'is-surface' : from_test(sim.is_surface),
#         'sample-placement' : from_gen_fn(sim.get_stable_gen()),
#         'sample-grasp' : from_gen_fn(sim.get_grasp_gen()),
#         'grasp-command' : from_fn(sim.get_grasp_traj_fn()),
#         'motion-plan' : from_fn(motion_planner),
#         # 'motion-plan-carry' : from_fn(carry_planner),
#         'forward-kinematics' : from_fn(sim.arm_fk),
#         'inverse-kinematics' : from_fn(sim.arm_ik),
#     }


#     problem=PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)
#     return problem



def main():
    sim = MySimInterface(domain_filename="chessboard_domain.pddl")
    print(sim.state)

    objects = ['largebox', 'smallbox1', 'smallbox2', 'smallbox3', 'smallbox4']
    entities = ['hole', *objects]

    # 8x8 chessboard layout
    size = {
        'largebox': (2,2),
        'smallbox1': (1,1),
        'smallbox2': (1,1),
        'smallbox3': (1,1),
        'smallbox4': (1,1),
    }
    pose = {
        'largebox': (0,3), # lower left cover; largebox is taking up [(0,3), (0,4), (1,3), (1,4)]
        'smallbox1': (6,2),
        'smallbox2': (4,5),
        'smallbox3': (7,1),
        'smallbox4': (5,5),
    }
    hole_tiles = [(2,2), (2,3), (3,2), (3,3)]
    is_hole = {
        (row, column):((row,column) in hole_tiles ) for row, column in itertools.product(range(8), range(8))
    }

    predicates = []
    qinit = sim.get_arm_config()
    predicates.extend([('AtArmConf', qinit)])
    predicates.extend([('AtPose', obj, *pose[obj]) for obj in objects])
    predicates.extend([('Size', obj, *size[obj]) for obj in objects])
    occupying = lambda p, s: itertools.product(p[0]+np.arange(s[0]), p[1]+np.arange(s[1]))
    for obj in objects:
        predicates.extend([('Occupying', obj, *p) for p in occupying(pose[obj], size[obj])])
    predicates.extend([('IsHole', *tile) for tile in hole_tiles])

    sim.set_state_variables(*predicates, assume_binary_relations=True)

    print(sim.state)


if __name__=='__main__':
    main()