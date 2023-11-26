(define (domain chessboard)
    (:requirements :strips :equality :negative-preconditions :derived-predicates)

    (:predicates
        (Entity ?e)
        (Obj ?o)
        (Robot ?r)
        (Pose ?p)
        (ArmConf ?q)
        (Tile ?t)

        (IsHole ?t)
        (Size ?e ?s)
        (Heading ?theta)
        (TileInfo ?p ?x ?y)
        (PoseInfo ?p ?t ?theta) ; maybe calculate by stream? or just be derived
        (AtPose ?e ?p)
        (AtHeading ?e ?h)
        (AtTile ?e ?t)
        (AtArmConf ?q)

        (Movable ?o)
        (On ?o ?t)
        (Occupying ?o ?t) ; calculate by stream
        (NextTo ?e ?t) ; calculate by stream
        (Facing ?e ?t) ; calculate by stream

        (ValidTile ?t) ; calculate by stream checking if within [0,7]x[0,7]
    )

    (:action push
        :parameters (?r ?o ?rt ?rh ?ot ?newtile)
        :precondition (and  (Robot ?r) (Obj ?o) (Heading ?rh) (Tile ?rt) (Tile ?ot)  (Tile ?newtile)
                            (Movable ?o) (AtTile ?r ?rt) (AtHeading ?r ?rh) (AtTile ?o ?ot)
                            (Facing ?rh ?ot) (NextTo ?rt ?ot) 
                            (NextTo ?newtile ?ot) (Facing ?rh ?newtile) (ValidTile ?newtile)
                    )
        :effect (and (AtTile ?r ?ot) (AtTile ?o ?newtile)
                    (not (AtTile ?r ?rt)) (not (AtTile ?o ?ot))
                )
    )
)