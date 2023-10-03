
from typing import List, Optional
import numpy as np

from examples.fetch.from_kuka_tamp.fetch_primitives import MyiGibsonSemanticInterface, Pose
from examples.fetch.from_kuka_tamp.utils.helpers import print_BodyGrasp, print_Command, print_Conf, print_bool
from examples.pybullet.utils.pybullet_tools.kuka_primitives import Command
from examples.pybullet.utils.pybullet_tools.utils import get_sample_fn


def _test__plan_holding_motion_stream(sim:MyiGibsonSemanticInterface, stream_map:dict, state:List, target:str):
    atpose_fluents = [fluent for fluent in state if fluent[0].lower()=='atpose' and fluent[1].lower() != target]
    print("AtPose Fluents: ", [(name, obj, tuple(sim._fmt_num_iter(p) for p in pose)) for name,obj,pose in atpose_fluents])
    grasp = next(stream_map['sample-grasp'](target))[0][0]

    # Trajectory = returning to home position after having grasped object
    # grasp_pose -> approach_pose -> home pose
    p1 = grasp.grasp_pose
    q1 = sim.arm_ik(*p1)
    p2 = (grasp.approach_pose[0], None)
    q2 = sim.arm_ik(*p2)
    # Home position
    joint_vector_indices = sim._motion_planner.robot_arm_indices
    q3 = tuple(np.array(sim._robot.untucked_default_joint_pos)[joint_vector_indices])
    p3 = sim.arm_fk(q3)

    # print(f"Start:  Config:  {sim._fmt_num_iter(q1)} \tPose:  {'  '.join(sim._fmt_num_iter(p) for p in p1 if p is not None)}")
    # print(f"Goal:   Config:  {sim._fmt_num_iter(q2)} \tPose:  {'  '.join(sim._fmt_num_iter(p) for p in p2 if p is not None)}")
    # # print("Goal config: ", sim._fmt_num_iter(q2))
    # stream = stream_map['plan-holding-motion'](q1, q2, grasp, atpose_fluents=atpose_fluents)
    # sample = next(stream)[0]
    # print(sample)
    # command = sample[0]
    # print_Command(sim, command)

    # print("\n")
    
    print(f"Start:  Config:  {sim._fmt_num_iter(q2)} \tPose:  {'  '.join(sim._fmt_num_iter(p) for p in p2 if p is not None)}")
    print(f"Goal:   Config:  {sim._fmt_num_iter(q3)} \tPose:  {'  '.join(sim._fmt_num_iter(p) for p in p3 if p is not None)}")
    stream = stream_map['plan-holding-motion'](q2, q3, grasp, atpose_fluents=atpose_fluents)
    sample = next(stream)[0]
    print(sample)
    command = sample[0]
    print_Command(sim, command)

def _test__test_cfree_traj_pose(sim:MyiGibsonSemanticInterface, stream_map:dict, obstacle:str,
                                start_pose:Optional[Pose]=None, end_pose:Optional[Pose]=None):
    def get_command():
        q1 = sim.arm_ik(start_pose) if start_pose is not None else sim.get_arm_config()
        q2 = sim.arm_ik(end_pose)   if end_pose   is not None else (qrand:=get_sample_fn(sim.robot_id, sim._arm_joint_ids))()
        
        command_stream = stream_map['plan-free-motion'](q1,q2)
        command = next(command_stream)[0]
        if len(command) < 1:
            if end_pose is not None:
                raise RuntimeError("f'plan-free-motion' stream could not find a path for start and end poses {start_pose} and{end_pose}")
            else:
                for _ in range(1000):
                    q2 = qrand()
                    command_stream = stream_map['plan-free-motion'](q1,q2)
                    command = next(command_stream)[0]
                    if len(command) >= 1:
                        break
        command = command[0]
        return command
    
    command = get_command()
    assert isinstance(command,Command), f"command '{command}' is not of type Command"
    print_Command(sim, command)

    obstacle_pose = sim.get_pose(obstacle)
    test_stream = stream_map['test-cfree-traj-pose'](command, obstacle, obstacle_pose)
    result = next(test_stream)
    print(result)
    print_bool(result)

def _test__test_cfree_approach_pose(sim:MyiGibsonSemanticInterface, stream_map:dict, target:str):
    b1 = target
    p1 = sim.get_pose(b1)

    grasp_stream = stream_map['sample-grasp'](b1)
    g1 = next(grasp_stream)[0][0]
    print_BodyGrasp(sim, g1)

    test = stream_map['test-cfree-approach-pose']
    for b2 in sim.objects:
        p2 = sim.get_pose(b2)
        stream = test(b1,p1,g1,b2,p2)
        result = next(stream)
        print_bool(result)

def _test__plan_free_motion_stream(sim:MyiGibsonSemanticInterface, stream_map:dict, state:List, target:str):
    atpose_fluents = [fluent for fluent in state if fluent[0].lower()=='atpose']
    print("AtPose Fluents: ", [(name, obj, tuple(sim._fmt_num_iter(p) for p in pose)) for name,obj,pose in atpose_fluents])
    grasp = next(stream_map['sample-grasp'](target))[0][0]

    q1 = sim.get_arm_config()
    p1 = sim.get_pose(sim.eef_id)
    p2 = (grasp.approach_pose[0], None)
    q2 = sim.arm_ik(*p2)
    p3 = grasp.grasp_pose
    q3 = sim.arm_ik(*p3)

    print(f"Start:  Config:  {sim._fmt_num_iter(q1)} \tPose:  {'  '.join(sim._fmt_num_iter(p) for p in p1 if p is not None)}")
    print(f"Goal:   Config:  {sim._fmt_num_iter(q2)} \tPose:  {'  '.join(sim._fmt_num_iter(p) for p in p2 if p is not None)}")
    # print("Goal config: ", sim._fmt_num_iter(q2))
    stream = stream_map['plan-free-motion'](q1, q2, atpose_fluents=atpose_fluents)
    command = next(stream)[0][0]
    print_Command(sim, command)

    print("\n")
    
    print(f"Start:  Config:  {sim._fmt_num_iter(q2)} \tPose:  {'  '.join(sim._fmt_num_iter(p) for p in p2 if p is not None)}")
    print(f"Goal:   Config:  {sim._fmt_num_iter(q3)} \tPose:  {'  '.join(sim._fmt_num_iter(p) for p in p3 if p is not None)}")
    stream = stream_map['plan-free-motion'](q2, q3, atpose_fluents=atpose_fluents)
    command = next(stream)[0][0]
    print_Command(sim, command)

def _test__sample_pose_stream(stream_map:dict, target:str, surface:str):
    stream = stream_map['sample-pose'](target, surface)
    for _ in range(10):
        sample = next(stream)[0]
        pos, orn = sample.pose
        print(f"\nPosition:   \t{tuple(pos)}\nOrientation:   \t{tuple(orn)}")

def _test__test_cfree_pose_pose(stream_map:dict, b1:str, b2:str):
    stream = stream_map['test-cfree-pose-pose'](b1, body2=b2)
    print_bool(next(stream)[0])
    
def _test__sample_grasp_stream(sim:MyiGibsonSemanticInterface, stream_map:dict, target:str):
    stream = stream_map['sample-grasp'](target)
    for _ in range(10):
        grasp = next(stream)[0][0]
        print_BodyGrasp(sim, grasp)

def _test__ik_stream(sim, stream_map, target):
    grasp = next(stream_map['sample-grasp'](target))[0][0]
    print_BodyGrasp(sim, grasp)
    stream = stream_map['inverse-kinematics'](target, grasp)
    for _ in range(1):
        sample = next(stream)[0]
        if not sample:
            print(f"IK failed: {sample}")
            return
        config, command = sample
        print_Conf(sim, config)
        print_Command(sim, command)

        # print(f"\nConfiguration:   \t{[round(q,3) for q in config]}")
        # print(
        #     f"\nCommand <{command}>:"
        # )
        # for i,cmd in enumerate(command.body_paths):
        #     prefix = f"  {i}) {cmd.__class__.__name__}"
        #     if isinstance(cmd,BodyPath):
        #         string = f"\n\t - Body: {sim.get_name(cmd.body)}" \
        #                  f"\n\t - {len(cmd.path)} waypoints" \
        #                  f"\n\t - {len(cmd.joints)} joints" \
        #                  f"\n\t - Attachments: {[sim.get_name(g.body) for g in cmd.attachments]}"
        #     elif isinstance(cmd, Attach):
        #         string = f"\n\t - Robot: {sim.get_name(cmd.robot)}" \
        #                  f" (link={cmd.link})" \
        #                  f"\n\t - Target: {sim.get_name(cmd.body)}" \

                
        #     print(prefix+string)




def _temp_test(sim, target):
    q_start = sim.get_arm_config()
    print("Initial arm config: ", sim._fmt_num_iter(q_start))
    print("Direct-to-target config: ", sim._fmt_num_iter(sim.arm_ik(*sim.get_pose(target))))

    grasp_gen = sim.get_grasp_gen()(target)

    path = [None,None]
    while not all(path):
        q_approach = q_grasp = None
        while not all((q_approach, q_grasp)):
            grasp = next(grasp_gen)[0]
            print_BodyGrasp(sim, grasp)
            q_approach = sim.arm_ik(*grasp.approach_pose)
            q_grasp = sim.arm_ik(*grasp.grasp_pose)
        print("Approach config: ", sim._fmt_num_iter(q_approach))
        print("Grasp config:", sim._fmt_num_iter(q_grasp))

        sim.set_arm_config(q_start)
        path[0] = sim._motion_planner.plan_arm_motion(q_approach)
        sim.set_arm_config(q_approach)
        path[1] = sim._motion_planner.plan_arm_motion(q_grasp)
    try:
        print("Start-to-Approach:")
        print([sim._fmt_num_iter(q) for q in path[0]])   
    except:
        print("Start-to-Approach:")
        print(path[0])
    try:
        print("Approach-to-Grasp:")
        print([sim._fmt_num_iter(q) for q in path[1]])   
    except:
        print("Approach-to-Grasp:")
        print(path[1])

    # return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

