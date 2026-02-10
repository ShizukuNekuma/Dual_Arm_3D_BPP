(define (problem dual-arm-dag-task)
        (:domain dual-arm-scheduling)
          (:objects
        flange_tool0 -manipulator
        box_1 box_2 -objs
        target_0 target_1 target_2 -pos
        pick_0 place_0 pick_1 place_1 pick_2 place_2 -task)
          (:init
    (arm-empty flange_tool0)

    (task-not-done pick_0)
    (task-not-done place_0)
    (task-not-done pick_1)
    (task-not-done place_1)
    (task-not-done pick_2)
    (task-not-done place_2)

    (precedes pick_0 place_0)
    (precedes place_0 pick_2)
    (precedes pick_1 place_1)
    (precedes pick_2 place_2)

    (is-pick-task pick_0 box_2)
    (is-place-task place_0 box_2 target_0)
    (is-pick-task pick_1 box_1)
    (is-place-task place_1 box_1 target_1)
    (is-pick-task pick_2 box_2)
    (is-place-task place_2 box_2 target_2)
  )
          (:goal (and
    (task-done place_0)
    (task-done place_1)
    (task-done place_2)
  ))
        )