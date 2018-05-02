(define (domain pick-and-place)
  (:requirements :strips :equality)
  (:predicates
    (Conf ?q)
    (Block ?b)
    (Pose ?b ?p)
    (Region ?r)
    (Kin ?b ?q ?p)
    (AtPose ?b ?p)
    (AtConf ?q)
    (Holding ?b)
    (HandEmpty)
    (CFree ?p1 ?p2)
    (Unsafe ?b ?p)
    (CanMove)
    (Contained ?b ?p ?r)
    (In ?b ?r)
  )
  (:functions
    (Distance ?q1 ?q2)
  )
  (:action move
    :parameters (?q1 ?q2)
    :precondition (and (Conf ?q1) (Conf ?q2)
                       (AtConf ?q1) (CanMove))
    :effect (and (AtConf ?q2)
                 (not (AtConf ?q1)) (not (CanMove))
             (increase (total-cost) (Distance ?q1 ?q2)))
  )
  (:action pick
    :parameters (?b ?p ?q)
    :precondition (and (Kin ?b ?q ?p)
                       (AtConf ?q) (AtPose ?b ?p) (HandEmpty))
    :effect (and (Holding ?b) (CanMove)
                 (not (AtPose ?b ?p)) (not (HandEmpty))
                 (increase (total-cost) 1))
  )
  (:action place
    :parameters (?b ?p ?q)
    :precondition (and (Kin ?b ?q ?p)
                       (AtConf ?q) (Holding ?b) (not (Unsafe ?b ?p)))
    :effect (and (AtPose ?b ?p) (HandEmpty) (CanMove)
                 (not (Holding ?b))
                 (increase (total-cost) 1))
  )
  (:derived (Unsafe ?b1 ?p1)
    (exists (?b2 ?p2) (and (Pose ?b1 ?p1) (Pose ?b2 ?p2) (not (CFree ?p1 ?p2))
                            (AtPose ?b2 ?p2)))
  )
  (:derived (In ?b ?r)
    (exists (?p) (and (Pose ?b ?p) (Region ?r) (Contained ?b ?p ?r)
                            (AtPose ?b ?p)))
  )
)