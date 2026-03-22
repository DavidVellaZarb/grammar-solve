(define (domain depot)
  (:requirements :typing)
  (:types place locatable - object
          depot distributor - place
          truck hoist surface - locatable
          pallet crate - surface)

  (:predicates (at ?x - locatable ?y - place)
               (on ?x - crate ?y - surface)
               (in ?x - crate ?y - truck)
               (lifting ?x - hoist ?y - crate)
               (available ?x - hoist)
               (clear ?x - surface))

  (:action drive
   :parameters (?x - truck ?from - place ?to - place)
   :precondition (and (at ?x ?from))
   :effect (and (not (at ?x ?from)) (at ?x ?to)))

  (:action lift
   :parameters (?x - hoist ?y - crate ?z - surface ?p - place)
   :precondition (and (at ?x ?p) (available ?x) (at ?y ?p) (on ?y ?z) (clear ?y))
   :effect (and (not (at ?y ?p)) (lifting ?x ?y) (not (clear ?y)) (not (available ?x))
                (not (on ?y ?z)) (clear ?z)))

  (:action drop
   :parameters (?x - hoist ?y - crate ?z - surface ?p - place)
   :precondition (and (at ?x ?p) (at ?z ?p) (clear ?z) (lifting ?x ?y))
   :effect (and (not (lifting ?x ?y)) (at ?y ?p) (clear ?y) (available ?x)
                (on ?y ?z) (not (clear ?z))))

  (:action load
   :parameters (?x - hoist ?y - crate ?z - truck ?p - place)
   :precondition (and (at ?x ?p) (at ?z ?p) (lifting ?x ?y))
   :effect (and (not (lifting ?x ?y)) (in ?y ?z) (available ?x)))

  (:action unload
   :parameters (?x - hoist ?y - crate ?z - truck ?p - place)
   :precondition (and (at ?x ?p) (at ?z ?p) (in ?y ?z) (available ?x))
   :effect (and (not (in ?y ?z)) (not (available ?x)) (lifting ?x ?y)))
)
