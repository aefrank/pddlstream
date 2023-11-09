(define (domain fetch-basic)
    (:requirements :strips :equality :negative-preconditions :derived-predicates)
    (:predicates
        ; types
        (Any ?x)
        (Obj ?o)
        (Sink ?s)
        (Stove ?s)
        (Floors ?f)
        (Ceilings ?c)
        (Walls ?w)
        (Celery ?c)
        (Radish ?r)

        (Conf ?q)
        (Pose ?p)
        (Grasp ?g)
        (Path ?p)    
        (Command ?c)

        ; constant properties
        (FK ?q ?p) ; forward kinematics; configuration q produces eef pose p

        (Movable ?o)  
        (Surface ?s)
        (Stackable ?o ?s)
        (Placement ?p ?o ?s)

        (PathInfo ?path ?q1 ?q2)
        (GraspInfo ?g ?o ?gp ?ap)

        (GraspCommand ?cmd ?g)
        (CommandStartConf ?q ?cmd)
        (CommandEndConf ?q ?cmd)

        ; state
        (AtConf ?q)
        (EefAtPose ?p)
        (EefAtPathStart ?path)
        (AtPose ?o ?p)
        
        (Grasping ?o ?g)
        

        (Cooked ?o)
        (Cleaned ?o)
        
        ; complex
        (AvailablePath ?path ?q1 ?q2)
        (AvailableCarry ?path ?o ?g)
        (CollisionFree ?path)

        ; derived
        (Holding ?o)
        (HandEmpty)
        (On ?p ?s)

        (PathStartPose ?p ?path)
        (PathEndPose ?p ?path)

        (GraspObj ?g ?o)
        (GraspApproachPose ?g ?ap)
        (GraspPose ?g ?gp)
        (GraspPath ?g ?path)

        (GraspObjCommand ?cmd ?o)

        
    )

    (:derived (Holding ?o)
        (exists (?g) (and (Obj ?o) (Grasp ?g) (Grasping ?o ?g)))
    )
    (:derived (HandEmpty)
        (not (exists (?o) (and (Obj ?o) (Holding ?o))))
    )
    (:derived (Stackable ?o ?s)
        (and (Movable ?o) (Surface ?s))
    )
    ; (:derived (On ?o ?s)
    ;     (and (Stackable ?o ?s) (exists (?p) (and (Pose ?p) (AtPose ?o ?p) (Placement ?p ?o ?s))))
    ; )
    (:derived (On ?o ?s)
          (exists (?p) (and (Obj ?o) (Surface ?s) (Pose ?p) (Placement ?p ?o ?s) (AtPose ?o ?p)))
    )
    (:derived (GraspObj ?g ?o)
        (exists (?ap ?gp) (and (Grasp ?g) (Obj ?o) (Pose ?ap) (Pose ?gp) (GraspInfo ?g ?o ?ap ?gp)))
    )
    (:derived (GraspApproachPose ?g ?ap)
        (exists (?o ?gp) (and (Grasp ?g) (Obj ?o) (Pose ?ap) (Pose ?gp) (GraspInfo ?g ?o ?ap ?gp)))
    )
    (:derived (GraspPose ?g ?gp)
        (exists (?o ?ap) (and (Grasp ?g) (Obj ?o) (Pose ?ap) (Pose ?gp) (GraspInfo ?g ?o ?ap ?gp)))
    )
    (:derived (GraspPath ?g ?path)
        (and (Grasp ?g) (Path ?path)
            (exists (?ap ?gp ?aq ?gq) (and 
                (GraspApproachPose ?g ?ap) (FK ?aq ?ap) 
                (GraspPose ?g ?gp)  (FK ?gq ?gp) 
                (PathInfo ?path ?aq ?gq) 
        )))
    )
    (:derived (EefAtPose ?p)
        (exists (?q) (and (Conf ?q) (AtConf ?q) (FK ?q ?p)) )
    )
    (:derived (EefAtPathStart ?path)
        (and 
            (Path ?path)
            (exists (?p) (and 
                (Pose ?p) (EefAtPose ?p) 
                (PathStartPose ?path ?p)) 
        ))
    )
    (:derived (PathStartPose ?p ?path)
        (and
            (Path ?path) (Pose ?p)
            (exists (?q1 ?q2) (and 
                (Conf ?q1) (Conf ?q2) (FK ?q1 ?p)
                (PathInfo ?path ?q1 ?q2)) ))
    )
    (:derived (PathEndPose ?p ?path)
        (and
            (Path ?path) (Pose ?p)
            (exists (?q1 ?q2) (and 
                (Conf ?q1) (Conf ?q2) (FK ?q2 ?p)
                (PathInfo ?path ?q1 ?q2)) ))
    )
    (:derived (GraspObjCommand ?cmd ?o)
        (and (Obj ?o) (Command ?cmd)
            (exists (?g) (and (GraspObj ?g ?o) (GraspCommand ?cmd ?g)))
        )
    )


    
    (:action move
        :parameters (?q1 ?q2 ?path)
        :precondition   (and    (Conf ?q1) (Conf ?q2) (Path ?path)
                                (AtConf ?q1) (not (AtConf ?q2)) (HandEmpty) 
                                (PathInfo ?path ?q1 ?q2) (CollisionFree ?path)
                        )
        :effect (and (AtConf ?q2)
                     (not (AtConf ?q1)) )   
    )

    ; (:action carry
    ;     :parameters (?q1 ?q2 ?path ?o ?g)
    ;     :precondition (and (Conf ?q1) (Conf ?q2) (Path ?path) (Obj ?o) (Grasp ?g) 
    ;                         (Movable ?o) (Grasping ?o ?g)
    ;                         (AtConf ?q1) (not (AtConf ?q2))
    ;                         (AvailablePath ?path ?q1 ?q2) (AvailableCarry ?path ?o ?g) )
    ;     :effect (and (AtConf ?q2) (not (AtConf ?q1)) )
    ; )

    (:action pick
        :parameters (?o ?g ?p ?cmd)
        :precondition (and  (GraspObj ?g ?o) (Pose ?p) (GraspCommand ?cmd ?g)
                            (Movable ?o) (HandEmpty) ; implies (not (Grasping ?o ?g))
                            (exists (?q) (and   (Conf ?q) (FK ?q ?p) 
                                                (CommandStartConf ?q ?cmd) (AtConf ?q)) )
                        ) 
        :effect (and (Grasping ?o ?g) (AtPose ?o ?p))
    )


    ; (:action place
    ;     :parameters (?o ?g ?s ?p ?cmd)
    ;     :precondition (and  (GraspObj ?o) (Grasp ?g) (Surface ?s) (Pose ?p) 
    ;                         (Grasping ?o ?g) (Placement ?p ?o ?s)
    ;                         (EefAtPathStart ?path) (PathEndPose ?path ?p)
    ;                     )
    ;     :effect (and (AtPose ?o ?p) (not (Grasping ?o ?p)))
    ; )

    (:action clean
        :parameters (?o ?s)
        :precondition (and (Obj ?o) (Sink ?s) (Stackable ?o ?s)
                        (On ?o ?s) (not (Cleaned ?o)) )
        :effect (Cleaned ?o)
    )

    (:action cook
        :parameters (?o ?s)
        :precondition (and (Obj ?o) (Stove ?s) (Stackable ?o ?s)
                        (On ?o ?s) (Cleaned ?o) )
        :effect (and (Cooked ?o)
                 (not (Cleaned ?o)))
    )
)