"""Chain cases. Classic: CLRS 6-matrix instance, also used in the paper.
Random chains: fixed seed, n in 4..8 matrices, dims from {8,16,...,128}."""
import random
CLASSIC = [30, 35, 15, 5, 10, 20, 25]
def cases():
    yield "clrs6", CLASSIC
    yield "clrs6x4", [4 * d for d in CLASSIC]
    yield "ex3", [20, 2, 30, 3]
    rng = random.Random(2027)
    for n in range(4, 9):
        for rep in range(3):
            yield f"rand{n}_{rep}", [rng.choice(range(8, 129, 8)) for _ in range(n + 1)]
