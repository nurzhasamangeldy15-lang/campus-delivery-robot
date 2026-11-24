"""Experiments for the report:  python evaluate.py   (fixed seed -> reproducible)"""
import json, random
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from campus_robot.world import WORLD
from campus_robot.hmm import forward_filter, STAY_PROB, P_EXACT, P_NEIGHBOR
from campus_robot.bayes import PRIOR, LIKELIHOOD, bayes_update
from campus_robot.astar import a_star, bfs
from campus_robot.mdp import value_iteration, decide, switch_probability, R, T, GAMMA

SEED, TRIALS = 42, 3000
man = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])


def sample_reading(rng, s):
    nb = WORLD.neighbors(s)
    x = rng.random()
    if x < P_EXACT:
        return s
    if x < P_EXACT + P_NEIGHBOR:
        return rng.choice(nb)
    return rng.choice([c for c in WORLD.free if c != s and c not in nb])


def hmm_experiment():
    rng, out = random.Random(SEED), {}
    for k in (1, 2, 3, 4, 5):
        hit = {"last reading": 0, "majority vote": 0, "HMM filter": 0}
        err = {m: 0 for m in hit}; conf = 0
        for _ in range(TRIALS):
            s = rng.choice(WORLD.free); obs = []
            for t in range(k):
                if t and rng.random() > STAY_PROB:
                    s = rng.choice(WORLD.neighbors(s))
                obs.append(sample_reading(rng, s))
            belief = forward_filter(WORLD, obs)
            guess = {"last reading": obs[-1],
                     "majority vote": Counter(obs).most_common(1)[0][0],
                     "HMM filter": max(belief, key=belief.get)}
            conf += max(belief.values())
            for m, g in guess.items():
                hit[m] += g == s; err[m] += man(g, s)
        out[k] = {m: {"accuracy": hit[m] / TRIALS, "mean_error": err[m] / TRIALS} for m in hit}
        out[k]["HMM mean confidence"] = conf / TRIALS
    return out


def search_experiment():
    stats = {n: [0, 0, 0] for n in ("BFS", "UCS (A* with h=0)", "A* (Manhattan)")}   # nodes, optimal, pairs
    pairs = [(s, g) for s in WORLD.free for g in WORLD.free if s != g]
    for s, g in pairs:
        best = len(bfs(WORLD, s, g)[0])
        for name, (p, e) in (("BFS", bfs(WORLD, s, g)), ("UCS (A* with h=0)", a_star(WORLD, s, g, False)),
                             ("A* (Manhattan)", a_star(WORLD, s, g))):
            stats[name][0] += e; stats[name][1] += len(p) == best; stats[name][2] += 1
    return {n: {"mean_expanded": v[0] / v[2], "optimal_rate": v[1] / v[2], "pairs": v[2]} for n, v in stats.items()}


def bayes_experiment():
    rng, hit, n = random.Random(SEED), 0, 5000
    for _ in range(n):
        room = rng.choice(list(PRIOR)); post = dict(PRIOR)
        for clue in LIKELIHOOD:
            post = bayes_update(post, clue, observed=rng.random() < LIKELIHOOD[clue][room])
        hit += max(post, key=post.get) == room
    return {"trials": n, "map_accuracy_with_2_clues": hit / n, "random_guess": 1 / 3}


def mdp_experiment():
    res = {}
    for pen in (-2, -8, -15):
        r2 = {s: dict(a) for s, a in R.items()}; r2["Lost"]["Move"] = pen
        V, _ = value_iteration(R=r2)
        res[f"penalty {pen}"] = {"switch_probability": switch_probability(V, R=r2),
                                 "decision_at_25%": decide(0.25, V, R=r2)[0],
                                 "decision_at_97%": decide(0.97, V, R=r2)[0]}
    return res


if __name__ == "__main__":
    res = {"seed": SEED, "trials": TRIALS, "hmm": hmm_experiment(), "search": search_experiment(),
           "bayes": bayes_experiment(), "mdp": mdp_experiment()}
    json.dump(res, open("results/results.json", "w"), indent=2)
    ks = list(res["hmm"])
    plt.figure(figsize=(6.4, 4))
    for m in ("last reading", "majority vote", "HMM filter"):
        plt.plot(ks, [res["hmm"][k][m]["accuracy"] for k in ks], marker="o", label=m)
    plt.xlabel("number of 2GIS readings"); plt.ylabel("position accuracy"); plt.ylim(0.5, 1)
    plt.legend(); plt.grid(alpha=.3); plt.title("Localisation accuracy"); plt.tight_layout()
    plt.savefig("results/hmm_accuracy.png", dpi=130)
    plt.figure(figsize=(6.4, 4))
    names = list(res["search"]); plt.bar(names, [res["search"][n]["mean_expanded"] for n in names],
                                         color=["#8FA0C4", "#F2A541", "#2A9D8F"])
    plt.ylabel("mean expanded nodes"); plt.title("Search effort (all cell pairs)"); plt.tight_layout()
    plt.savefig("results/search_nodes.png", dpi=130)
    print(json.dumps(res, indent=1))
