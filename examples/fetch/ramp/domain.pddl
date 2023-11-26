(define (domain fetch-ramp)
    (:requirements :strips :equality :negative-preconditions :derived-predicates)
    (:predicates
        ; types
        (Obj ?o)
        (Floors ?f)
        (Ceilings ?c)
        (Walls ?w)

        (Pose ?p)
        (BasePose ?bp)
        (ArmConf ?q)
        (ArmPath ?apath)    
        (BasePath ?bpath)

        ; constant properties
        (ArmConfPoseEq ?q ?p) ; forward kinematics; configuration q produces eef pose p

        (Movable ?o)  
        (Surface ?s)
        (Stackable ?o ?s)
        (Placement ?p ?o ?s)

        
        (ArmPathInfo ?apath ?q1 ?q2)
        (ArmPathStartConf ?q ?apath) ; derived
        (ArmPathEndConf ?q ?apath) ; derived
        (ArmPathStartPose ?p ?apath) ; derived
        (ArmPathEndPose ?p ?apath) ; derived

        (BasePathInfo ?apath ?p1 ?p2)
        (BasePathStartPose ?p ?bpath) ; derived
        (BasePathEndPose ?p ?bpath) ; derived

        ; state
        (AtArmConf ?q)
        (EefAtPose ?p)
        (EefAtPathStart ?path)
        (AtPose ?o ?p)
        
        (InReach ?q ?p)
        (Holding ?o)
        (HandEmpty)

        (On ?o ?s)

        ; complex
        (CollisionFree ?path)

    )

    (:derived (HandEmpty)
        (not (exists (?o) (and (Obj ?o) (Holding ?o))))
    )
    (:derived (Stackable ?o ?s)
        (and (Movable ?o) (Surface ?s))
    )
    (:derived (On ?o ?s)
          (exists (?p) (and (Obj ?o) (Surface ?s) (Pose ?p) (Placement ?p ?o ?s) (AtPose ?o ?p)))
    )
    (:derived (EefAtPose ?p)
        (exists (?q) (and (ArmConf ?q) (AtArmConf ?q) (ArmConfPoseEq ?q ?p)) )
    )
    (:derived (EefAtArmPathStart ?apath)
        (and 
            (ArmPath ?apath)
            (exists (?p) (and 
                (Pose ?p) (EefAtPose ?p) 
                (ArmPathStartPose ?apath ?p)) 
        ))
    )
    (:derived (ArmPathStartConf ?q ?apath)
        (and
            (ArmPath ?apath) (ArmConf ?q)
            (exists (?q2) (and (ArmConf ?q2) (ArmPathInfo ?apath ?q ?q2)) ))
    )
    (:derived (ArmPathEndConf ?q ?apath)
        (and
            (ArmPath ?apath) (ArmConf ?q)
            (exists (?q1) (and (ArmConf ?q1) (ArmPathInfo ?apath ?q1 ?q)) ))
    )
    (:derived (ArmPathStartPose ?p ?path)
        (and
            (ArmPath ?apath) (Pose ?p)
            (exists (?q1 ?q2) (and 
                (ArmConf ?q1) (ArmConf ?q2) (ArmConfPoseEq ?q1 ?p)
                (ArmPathInfo ?apath ?q1 ?q2)) ))
    )
    (:derived (ArmPathEndPose ?p ?apath)
        (and
            (ArmPath ?apath) (Pose ?p)
            (exists (?q1 ?q2) (and 
                (ArmConf ?q1) (ArmConf ?q2) (ArmConfPoseEq ?q2 ?p)
                (ArmPathInfo ?apath ?q1 ?q2)) ))
    )
    (:derived (BasePathStartPose ?p ?bpath)
        (and
            (BasePath ?bpath) (Pose ?p)
            (exists (?p2) (and (Pose ?p2) (BasePathInfo ?apath ?p ?p2)) ))
    )
    (:derived (BasePathEndPose ?p ?bpath)
        (and
            (BasePath ?bpath) (Pose ?p)
            (exists (?p1) (and (Pose ?p1) (BasePathInfo ?apath ?p1 ?p)) ))
    )


    
    (:action move_arm
        :parameters (?q1 ?q2 ?apath)
        :precondition   (and    (ArmConf ?q1) (ArmConf ?q2) (ArmPath ?apath)
                                (AtArmConf ?q1) (not (AtArmConf ?q2)) 
                                (ArmPathInfo ?apath ?q1 ?q2) (CollisionFree ?apath)
                                (HandEmpty) 
                        )
        :effect (and (AtArmConf ?q2)
                     (not (AtArmConf ?q1)) )   
    )

    (:action move_arm_carry
        :parameters (?q1 ?q2 ?apath ?o ?p2)
        :precondition   (and    (ArmConf ?q1) (ArmConf ?q2) (ArmPath ?apath)
                                (AtArmConf ?q1) (not (AtArmConf ?q2)) (ArmConfPoseEq ?q2 ?p2)
                                (ArmPathInfo ?apath ?q1 ?q2) (CollisionFree ?apath)
                                (Holding ?o) 
                        )
        :effect (and (AtArmConf ?q2) 
                     (not (AtArmConf ?q1)) 
                     (AtPose ?o ?p2)
                )   
    )

    (:action pick
        :parameters (?o ?p ?apath ?q1 ?q2)
        :precondition (and  (Obj ?o) (Pose ?p) (ArmPath ?apath) (ArmConf ?q1) (ArmConf ?q2)
                            (Movable ?o) 
                            (ArmPathInfo ?apath ?q1 ?q2) (ArmConfPoseEq ?q2 ?p) (CollisionFree ?apath) 
                            (HandEmpty) (AtPose ?o ?p) (AtArmConf ?q1)
                        ) 
        :effect (and (Holding ?o) (AtPose ?o ?p))
    )

    (:action place
        :parameters (?o ?p ?s ?apath ?q1 ?q2)
        :precondition (and  (Obj ?o) (Surface ?s) (Pose ?p) (ArmPath ?apath) (ArmConf ?q1) (ArmConf ?q2)
                            (Movable ?o) (Placement ?p ?o ?s) 
                            (ArmPathInfo ?apath ?q1 ?q2) (ArmConfPoseEq ?q2 ?p) (CollisionFree ?apath) 
                            (AtArmConf ?q1) (Holding ?o) 
                        )
        :effect (and (AtPose ?o ?p) (not (Holding ?o)))
    )

)