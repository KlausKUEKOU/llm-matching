"""Many-to-one matching: student-proposing deferred acceptance and stability checks."""


def deferred_acceptance(student_prefs, program_priorities, capacities):
    """Gale-Shapley (1962) student-proposing deferred acceptance.

    student_prefs: {student: [program, ...]} most preferred first
    program_priorities: {program: [student, ...]} highest priority first
    capacities: {program: int}
    Returns {program: [students]} — stable and strategy-proof for students.
    """
    rank = {p: {s: i for i, s in enumerate(order)} for p, order in program_priorities.items()}
    next_choice = {s: 0 for s in student_prefs}
    held = {p: [] for p in program_priorities}
    free = list(student_prefs)
    while free:
        s = free.pop(0)
        prefs = student_prefs[s]
        if next_choice[s] >= len(prefs):
            continue                                   # exhausted list: stays unmatched
        p = prefs[next_choice[s]]
        next_choice[s] += 1
        held[p].append(s)
        held[p].sort(key=lambda x: rank[p][x])
        if len(held[p]) > capacities[p]:
            free.append(held[p].pop())                 # reject the lowest-priority student
    return held


def assignment_of(matching):
    return {s: p for p, studs in matching.items() for s in studs}


def blocking_pairs(matching, true_prefs, program_priorities, capacities):
    """Pairs (s, p) where s prefers p to her assignment and p has a free seat or holds someone
    of lower priority than s. Zero blocking pairs = stable (no justified envy)."""
    assign = assignment_of(matching)
    rank = {p: {s: i for i, s in enumerate(order)} for p, order in program_priorities.items()}
    pairs = []
    for s, prefs in true_prefs.items():
        current = assign.get(s)
        cutoff = prefs.index(current) if current in prefs else len(prefs)
        for p in prefs[:cutoff]:
            held = matching[p]
            if len(held) < capacities[p] or any(rank[p][s] < rank[p][x] for x in held):
                pairs.append((s, p))
    return pairs


def is_feasible(matching, students, capacities):
    assigned = [s for studs in matching.values() for s in studs]
    return (all(len(matching[p]) <= capacities[p] for p in matching)
            and len(assigned) == len(set(assigned)) and set(assigned) <= set(students))
