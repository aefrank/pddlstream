(define (stream fetch-basic)
    (:stream sample-conf
        :inputs (?x)
        :domain (Any ?x)
        :outputs (?q)
        :certified (Conf ?q)
    )
    (:stream motion-plan
        :inputs (?q1 ?q2)
        :domain (and (Conf ?q1) (Conf ?q2))
        :outputs (?path)
        :certified (and (Path ?path) (CollisionFree ?path))
    )
)