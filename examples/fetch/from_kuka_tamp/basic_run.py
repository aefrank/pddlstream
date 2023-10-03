


import os
from examples.fetch.from_kuka_tamp.fetch_primitives import MyiGibsonSemanticInterface
from examples.pybullet.utils.pybullet_tools.kuka_primitives import BodyConf, BodyPath
from pddlstream.algorithms.meta import create_parser, solve
from pddlstream.language.constants import PDDLProblem
from pddlstream.language.generator import from_fn
from pddlstream.utils import Profiler, read




def main():
    parser = create_parser()
    parser.add_argument('-g','--gui', action='store_true', help='Render and visualize the system')
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = "fetch_tamp.yaml"
    config_path = os.path.join(dir_path,config)

    sim = MyiGibsonSemanticInterface(
        config_file=config_path, 
        # objects=objects,
        headless=(not args.gui)
    )

    qinit = sim.get_arm_config()
    qgoal = sim.sample_arm_config()

    print(qinit)
    print(qgoal)

    plan = sim.plan_arm_joint_motion(qinit, qgoal)
    print('motion plan length:', len(plan))
    print('\n\n\n')


    domain_pddl = read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'toy_domain.pddl'))
    constant_map = {}
    stream_pddl =  read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'toy_stream.pddl'))
    motion_plan_fn = lambda q1, q2: (BodyPath(sim.robot_bid, sim.plan_arm_joint_motion(q1.configuration,q2.configuration)),)
    stream_map = {
        'sample-conf' : from_fn(sim.sample_arm_config),
        'motion-plan' : from_fn(motion_plan_fn)
    }
    q1 = BodyConf(sim.robot_bid, qinit)
    q2 = BodyConf(sim.robot_bid, qgoal)
    init = [("Conf", q1), ("AtConf", q1), ("Conf", q2)]
    goal = ("AtConf", q2)
    problem = PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

    # with Profiler():
    # solution = solve(problem, unit_costs=True)
    planner = 'lmcut-astar'
    solution = solve(problem, unit_costs=True, planner=planner, debug=True)
    print(solution)







if __name__=='__main__':
    main()