"""Bayesian inference: which candidate room is the target?"""

CANDIDATES = {"A": (5, 0), "B": (0, 5), "C": (5, 3)}      # candidate classroom cells
PRIOR = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}

# P(clue is given | classroom really is X) - predefined assumptions
LIKELIHOOD = {
    "east side":  {"A": 0.10, "B": 0.80, "C": 0.30},
    "north part": {"A": 0.20, "B": 0.80, "C": 0.20},
}


def bayes_update(prior, clue, observed=True):
    """posterior(X) = prior(X) * P(clue|X) / P(clue)   (observed=False uses 1 - P(clue|X))."""
    if clue not in LIKELIHOOD:
        raise ValueError(f"Unknown clue '{clue}'. Known: {list(LIKELIHOOD)}")
    unnorm = {x: prior[x] * (LIKELIHOOD[clue][x] if observed else 1 - LIKELIHOOD[clue][x])
              for x in prior}
    evidence = sum(unnorm.values())                        # P(clue)
    if evidence == 0:
        raise ValueError("Clue has zero probability under every candidate")
    return {x: v / evidence for x, v in unnorm.items()}
