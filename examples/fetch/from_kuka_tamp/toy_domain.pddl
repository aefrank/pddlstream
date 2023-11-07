(define (domain fetch-basic)
    (:requirements :strips :equality :negative-preconditions)
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

        ; constant properties
        (Movable ?o)  
        (Surface ?s)
        (Stackable ?o ?s)
        (Placement ?p ?o ?s)

        ; state
        (AtConf ?q)
        (AtPose ?o ?p)
        (Grasping ?o ?g)
        (GraspForObj ?o ?g)
        (Cooked ?o)
        (Cleaned ?o)
        
        ; complex
        (ValidPath ?path ?q1 ?q2)
        (ValidCarry ?path ?o ?g)

        ; derived
        (Holding ?o)
        (HandEmpty)
        (On ?p ?s)
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

    (:action move
        :parameters (?q1 ?q2 ?p)
        :precondition   (and (Conf ?q1) (Conf ?q2) (Path ?p)
                            (HandEmpty)
                            (AtConf ?q1)
                            (not (AtConf ?q2))
                            (ValidPath ?p ?q1 ?q2)
                        )
        :effect (and (AtConf ?q2)
                     (not (AtConf ?q1)) )   
    )

    (:action carry
        :parameters (?q1 ?q2 ?path ?o ?g)
        :precondition (and (Conf ?q1) (Conf ?q2) (Path ?path) (Obj ?o) (Grasp ?g) 
                            (Movable ?o) (Grasping ?o ?g)
                            (AtConf ?q1) (not (AtConf ?q2))
                            (ValidPath ?path ?q1 ?q2) (ValidCarry ?path ?o ?g) )
        :effect (and (AtConf ?q2) (not (AtConf ?q1)) )
    )

    (:action pick
        :parameters (?o ?p ?g)
        :precondition (and  (Obj ?o) (Pose ?p) (Grasp ?g) 
                            (Movable ?o) (GraspForObj ?o ?g) 
                            (AtPose ?o ?p) (HandEmpty) ) ; implies (not (Grasping ?o ?g))
        :effect (and (not (AtPose ?o ?p)) (Grasping ?o ?g) )
    )

    (:action place
        :parameters (?o ?g ?s ?p)
        :precondition (and  (Obj ?o) (Grasp ?g) (Surface ?s) (Pose ?p) 
                            (GraspForObj ?o ?g) (Grasping ?o ?g) 
                            (Placement ?p ?o ?s)
                        )
        :effect (and (AtPose ?o ?p) (not (Grasping ?o ?p)))
    )

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