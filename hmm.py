"""HMM: where is the robot?  Hidden state = real cell, observation = cell reported by 2GIS."""

STAY_PROB = 0.8                         # robot is almost stationary while localising
P_EXACT, P_NEIGHBOR, P_FAR = 0.70, 0.24, 0.06   # 2GIS sensor model


def transition(world, s, s2):
    """P(next cell = s2 | current cell = s)."""
    nb = world.neighbors(s)
    if not nb:
        return 1.0 if s == s2 else 0.0
    if s2 == s:
        return STAY_PROB
    return (1 - STAY_PROB) / len(nb) if s2 in nb else 0.0


def emission(world, obs, s):
    """P(2GIS reports obs | robot really in s): exact 70%, neighbour 24%, anywhere else 6%."""
    nb = world.neighbors(s)
    if not nb:
        return P_EXACT + P_NEIGHBOR if obs == s else P_FAR / (len(world.free) - 1)
    if obs == s:
        return P_EXACT
    if obs in nb:
        return P_NEIGHBOR / len(nb)
    return P_FAR / (len(world.free) - 1 - len(nb))


def _check(world, observations):
    obs = [tuple(o) for o in observations]
    for o in obs:
        if not world.is_free(o):
            raise ValueError(f"2GIS reading {o} is not a walkable cell")
    return obs


def predict(world, belief):
    """Step 1 of the forward algorithm: the robot may have moved a little."""
    new = {s: 0.0 for s in world.free}
    for s, p in belief.items():
        nb = world.neighbors(s)
        if not nb:
            new[s] += p
            continue
        new[s] += p * STAY_PROB
        for s2 in nb:
            new[s2] += p * (1 - STAY_PROB) / len(nb)
    return new


def forward_filter(world, observations):
    """Belief over the robot's cell after all readings (predict -> update -> normalise)."""
    obs = _check(world, observations)
    belief = {s: 1 / len(world.free) for s in world.free}
    for o in obs:
        pred = predict(world, belief)
        new = {s: pred[s] * emission(world, o, s) for s in world.free}
        total = sum(new.values())
        belief = {s: v / total for s, v in new.items()}
    return belief


def viterbi(world, observations):
    """Most likely sequence of real cells given all readings."""
    obs = _check(world, observations)
    V = [{s: emission(world, obs[0], s) / len(world.free) for s in world.free}]
    back = []
    for o in obs[1:]:
        col, ptr = {}, {}
        for s2 in world.free:
            best = max([s2] + world.neighbors(s2), key=lambda s: V[-1][s] * transition(world, s, s2))
            col[s2] = V[-1][best] * transition(world, best, s2) * emission(world, o, s2)
            ptr[s2] = best
        m = max(col.values())
        V.append({s: v / m for s, v in col.items()})      # rescale to avoid underflow
        back.append(ptr)
    seq = [max(V[-1], key=V[-1].get)]
    for ptr in reversed(back):
        seq.append(ptr[seq[-1]])
    return list(reversed(seq))
