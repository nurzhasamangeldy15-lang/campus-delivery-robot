"""MDP: given how sure the robot is about its position, move now or re-localize first?
States: Localized / Lost.  Actions: Move / Relocalize.  Solved by value iteration."""

STATES = ["Localized", "Lost"]
ACTIONS = ["Move", "Relocalize"]
GAMMA = 0.9

T = {  # T[state][action] = {next_state: probability}
    "Localized": {"Move": {"Localized": 0.80, "Lost": 0.20},
                  "Relocalize": {"Localized": 0.95, "Lost": 0.05}},
    "Lost":      {"Move": {"Localized": 0.10, "Lost": 0.90},
                  "Relocalize": {"Localized": 0.70, "Lost": 0.30}},
}
R = {   # immediate reward
    "Localized": {"Move": +10, "Relocalize": -1},
    "Lost":      {"Move": -8,  "Relocalize": -1},
}


def q_value(s, a, V, R=R, T=T, gamma=GAMMA):
    """Q(s,a) = R(s,a) + gamma * sum_s' T(s,a,s') * V(s')"""
    return R[s][a] + gamma * sum(p * V[s2] for s2, p in T[s][a].items())


def value_iteration(R=R, T=T, gamma=GAMMA, tol=1e-9, max_iter=10_000):
    """Repeat the Bellman update V(s) = max_a Q(s,a) until values stop changing."""
    V = {s: 0.0 for s in STATES}
    for i in range(1, max_iter + 1):
        new = {s: max(q_value(s, a, V, R, T, gamma) for a in ACTIONS) for s in STATES}
        delta = max(abs(new[s] - V[s]) for s in STATES)
        V = new
        if delta < tol:
            break
    return V, i


def decide(p_localized, V, R=R, T=T, gamma=GAMMA):
    """Treat the HMM confidence as P(Localized); pick the action with the larger expected Q."""
    if not 0.0 <= p_localized <= 1.0:
        raise ValueError("p_localized must be in [0, 1]")
    expected = {a: p_localized * q_value("Localized", a, V, R, T, gamma)
                + (1 - p_localized) * q_value("Lost", a, V, R, T, gamma) for a in ACTIONS}
    return max(expected, key=expected.get), expected


def switch_probability(V, R=R, T=T, gamma=GAMMA):
    """Confidence p* above which Move beats Relocalize (both expected values are linear in p)."""
    q = lambda s, a: q_value(s, a, V, R, T, gamma)
    dm = q("Localized", "Move") - q("Lost", "Move")
    dr = q("Localized", "Relocalize") - q("Lost", "Relocalize")
    return (q("Lost", "Relocalize") - q("Lost", "Move")) / (dm - dr)
