"""Glue: HMM -> Bayes -> A* -> MDP in one call."""
from .world import WORLD
from .hmm import forward_filter, viterbi
from .bayes import PRIOR, CANDIDATES, bayes_update
from .astar import a_star
from .mdp import value_iteration, decide, switch_probability


def run(readings, clues, world=WORLD):
    if not readings:
        raise ValueError("Provide at least one 2GIS reading")
    belief = forward_filter(world, readings)                    # 1. where am I?
    best_cell = max(belief, key=belief.get)
    confidence = belief[best_cell]
    posterior, history = dict(PRIOR), [dict(PRIOR)]             # 2. where is the room?
    for clue in clues:
        posterior = bayes_update(posterior, clue)
        history.append(posterior)
    target_name = max(posterior, key=posterior.get)
    path, expanded = a_star(world, best_cell, CANDIDATES[target_name])   # 3. route
    V, _ = value_iteration()                                    # 4. move or re-check?
    action, expected = decide(confidence, V)
    return dict(readings=[tuple(r) for r in readings], clues=list(clues), belief=belief,
                best_cell=best_cell, confidence=confidence, viterbi=viterbi(world, readings),
                posterior=posterior, history=history, target_name=target_name,
                target=CANDIDATES[target_name], path=path, expanded=expanded,
                V=V, expected=expected, action=action, switch_p=switch_probability(V))
