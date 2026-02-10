(define (domain dual-arm-scheduling)
  (:requirements :strips :equality)
  (:predicates
    (arm-empty ?m -manipulator :discrete)
    (holding ?m -manipulator ?x -objs (kinematic (link ?m ?x) (unlink ?m ?x)))
    (task-done ?t -task :discrete)
    (precedes ?prev -task ?next -task :discrete)
    (is-pick-task ?t -task ?x -objs :discrete)
    (is-place-task ?t -task ?x -objs ?tgt -objs :discrete)
    (at ?m -manipulator ?x -objs)
  )

  (:action pickup
    :parameters (?m -manipulator ?o -objs ?t -task)
    :precondition (and 
      (not (task-done ?t))
      (forall (?p -task) 
        (imply (precedes ?p ?t) (task-done ?p)))
      (is-pick-task ?t ?o)
      (arm-empty ?m)
      (at ?m ?o) 
    )
    :effect (and 
      (task-done ?t)
      (not (arm-empty ?m))
      (holding ?m ?o)
    )
  )

  (:action putdown
    :parameters (?m -manipulator ?o -objs ?tgt -objs ?t -task)
    :precondition (and 
        (not (task-done ?t))
        (forall (?p -task) 
          (imply (precedes ?p ?t) (task-done ?p)))
        (is-place-task ?t ?o ?tgt)
        (holding ?m ?o)
        (at ?m ?tgt)
    )
    :effect (and 
      (task-done ?t)
      (not (holding ?m ?o))
      (arm-empty ?m)
    )
  )
)