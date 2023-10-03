from typing import Iterable, Union
from examples.fetch.from_kuka_tamp.utils.utils import is_numeric, recursive_map_advanced
from examples.pybullet.utils.pybullet_tools.kuka_primitives import Attach, BodyPath


def fmt_fluents(fluents):
    fn = lambda elem: round(elem,3) if is_numeric(elem) else elem
    fluent_str = "{"
    for fluent in fluents:
        label, *args = fluent
        args = recursive_map_advanced(fn, args, preserve_iterable_types=False, nonterminal_post_recursion_fn=tuple)
        fluent_str += str((label, *args)) + " "
    fluent_str += "}"
    return fluent_str

def print_bool(result:Union[bool,Iterable[bool]]) -> None:
    if result==[()]:
        print(f"Result: \tTrue")
    elif result==[]:
        print(f"Result: \tFalse")
    else:
        raise ValueError(f"Unexpected 'stream boolean' {result} of type {type(result)}. Expected [()]=True or []=False.")


def print_Conf(sim, conf):
    print(f"\nConfiguration:   \t{sim._fmt_num_iter(conf)}")

def print_BodyGrasp(sim, grasp):
    x,y,z = grasp.grasp_pose[0]
    a,b,c,d = grasp.grasp_pose[1]
    u,v,w = grasp.approach_pose[0]
    p,q,r,s = grasp.approach_pose[1]
    grasp_body = sim.get_name(grasp.body)
    grasp_robot = sim.get_name(grasp.robot)
    print(
        f"\nBody Grasp <{grasp}>:"
        f"\n  - Body:           name: {grasp_body}\tBID: {grasp.body}"
        f"\n  - Grasp Pose:     Position: ({x:.2f},{y:.2f},{z:.2f})    \tOrientation: ({a:.2f},{b:.2f},{c:.2f},{d:.2f}) "
        f"\n  - Approach pose:  Position: ({u:.2f},{v:.2f},{w:.2f})    \tOrientation: ({p:.2f},{q:.2f},{r:.2f},{s:.2f}) "
        f"\n  - Robot:          name: {grasp_robot}\tBID: {grasp.robot}"
        f"\n  - Link:           {grasp.link}"
        f"\n  - Index:          {grasp.index}"
    )

def print_Command(sim, command):
    print(
        f"\nCommand <{command}>:"
    )
    for i,cmd in enumerate(command.body_paths):
        prefix = f"  {i}) {cmd.__class__.__name__}"
        if isinstance(cmd,BodyPath):
            string = f"\n\t - Body: {sim.get_name(cmd.body)}" \
                        f"\n\t - {len(cmd.path)} waypoints" \
                        f"\n\t - {len(cmd.joints)} joints" \
                        f"\n\t - Attachments: {[sim.get_name(g.body) for g in cmd.attachments]}"
        elif isinstance(cmd, Attach):
            string = f"\n\t - Robot: {sim.get_name(cmd.robot)}" \
                        f" (link={cmd.link})" \
                        f"\n\t - Target: {sim.get_name(cmd.body)}" \

            
        print(prefix+string)