def simple_probability(m, n):
    poss = m / n
    return poss


def logical_or(m, k, n):
    poss = m / n + k / n
    return poss


def logical_and(m, k, n, l):
    poss = (m / n) * (k / l)
    return poss


def expected_value(values, probabilities):
    exp = 0
    for i in range(len(values)):
        exp += values[i] * probabilities[i]
    return exp


def conditional_probability(values):
    count_A = 0
    count_AB = 0

    for pair in values:
        n, m = pair
        if n == 1:
            count_A += 1
            if m == 1:
                count_AB += 1

    if count_A == 0:
        return 0

    poss = count_AB / count_A

    return poss


def bayesian_probability(a, b, ba):
    if b == 0:
        return 0
    poss = (ba * a) / b
    return poss






