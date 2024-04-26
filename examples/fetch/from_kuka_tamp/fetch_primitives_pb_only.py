from typing import Optional
import math

import pybullet as pb
from examples.fetch.from_kuka_tamp.fetch_primitives import JointPos
from examples.fetch.from_kuka_tamp.utils.object_spec import Orientation, Position
from iGibson.igibson.robots.fetch import Fetch
from utils import get_client, quat_from_euler

FETCH = Fetch()

def arm_ik(robot_bid, eef_position:Position, eef_orientation:Optional[Orientation]=None, 
           use_nullspace:bool=True, **kwargs) -> Optional[JointPos]:
    '''Arm Inverse Kinematics: Calculate Fetch arm joint configuration that satisfies the given end-effector position in the workspace (ignoring collisions). 
    Can additionally specify a target end-effector orientation and/or joint configuration nullspace.
    '''
    # with UndoableContext(self._robot):   
    *null_space, joint_damping = FETCH._motion_planner.get_ik_parameters()
    threshold = FETCH._motion_planner.arm_ik_threshold
    # ik_solver = pb.IK_DLS or pb.IK_SDLS
    ik_spec =  {
        # robot/eef body IDs
        'bodyUniqueId' : robot_bid, 
        'endEffectorLinkIndex' : FETCH.eef_links[FETCH.default_arm].link_id, 
        # workspace target
        'targetPosition' : eef_position, 
        # IK solver args
        'jointDamping' : joint_damping,
        #'solver' : ik_solver,
        'maxNumIterations' : 100,
        'residualThreshold' : threshold,
        # additional pybullet.calculateInverseKinematics kwargs
        'physicsClientId' : get_client(), 
        **kwargs
    }
    
    # Determine if result should be constrained by a target orientation or the Fetch arm's nullspace
    orientation_spec = (
        {} if eef_orientation is None else                                          # no eef orientation specified
        {'targetOrientation': eef_orientation} if len(eef_orientation)==4 else 
        {'targetOrientation': quat_from_euler(eef_orientation)}        
    )
    nullspace_spec = (
        {} if not use_nullspace else
        dict(zip(['upperLimits', 'lowerLimits', 'restPoses', 'jointRanges'], null_space)) \
    )
    
    # Calculate IK
    q = pb.calculateInverseKinematics(**ik_spec, **orientation_spec, **nullspace_spec)

    # Check if config is valid
    if (q is None) or any(map(math.isnan, q)):
        return None
    
    # Extract joint angles from full configuration
    joint_vector_indices = FETCH._motion_planner.robot_arm_indices
    q = tuple(np.array(q)[joint_vector_indices])

    return q