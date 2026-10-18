# Appariement plusieurs-à-un : acceptation différée (Gale-Shapley) et contrôle de stabilité.


# ------------------------------------------------------------------------------
# ALGORITHME AcceptationDifferee
# ENTREES : prefs_etudiants[s]  = liste des programmes, du préféré au moins aimé
#           priorites[p]        = liste des étudiants, du mieux classé au moins bien
#           capacites[p]        = nombre de places du programme p
# SORTIE  : retenus[p]          = liste des étudiants admis dans p
# DEBUT
#   POUR chaque programme p : retenus[p] <- liste vide
#   POUR chaque étudiant s : prochain[s] <- 0
#   libres <- tous les étudiants
#   TANT QUE libres n'est pas vide FAIRE
#       s <- premier étudiant de libres ; le retirer de libres
#       SI s a déjà candidaté partout ALORS passer au suivant
#       p <- programme numéro prochain[s] dans la liste de s
#       prochain[s] <- prochain[s] + 1
#       ajouter s dans retenus[p] et trier retenus[p] par priorité
#       SI taille(retenus[p]) > capacites[p] ALORS
#           rejeter le dernier de retenus[p] et le remettre dans libres
#   FIN TANT QUE
#   RETOURNER retenus
# FIN
# ------------------------------------------------------------------------------
def deferred_acceptance(student_prefs, program_priorities, capacities):
    held = {}
    for p in program_priorities:
        held[p] = []

    next_choice = {}
    for s in student_prefs:
        next_choice[s] = 0

    free = list(student_prefs)
    while len(free) > 0:
        s = free.pop(0)
        prefs = student_prefs[s]
        if next_choice[s] >= len(prefs):
            continue                                    # plus aucun programme à demander

        p = prefs[next_choice[s]]
        next_choice[s] = next_choice[s] + 1

        held[p].append(s)
        held[p] = sort_by_priority(held[p], program_priorities[p])

        if len(held[p]) > capacities[p]:
            rejected = held[p].pop()                    # le moins prioritaire est rejeté
            free.append(rejected)
    return held


def sort_by_priority(students, priority_list):
    """Trie une liste d'étudiants selon leur position dans la liste de priorité du programme."""
    ordered = []
    for s in priority_list:
        if s in students:
            ordered.append(s)
    return ordered


def assignment_of(matching):
    """Transforme {programme: [étudiants]} en {étudiant: programme}."""
    assign = {}
    for p in matching:
        for s in matching[p]:
            assign[s] = p
    return assign


# ------------------------------------------------------------------------------
# ALGORITHME PairesBloquantes
# Une paire (s, p) bloque l'appariement si :
#   s préfère p à son affectation actuelle
#   ET (p a une place libre OU p a admis quelqu'un de moins prioritaire que s)
# DEBUT
#   paires <- liste vide
#   POUR chaque étudiant s FAIRE
#       POUR chaque programme p que s préfère à son affectation FAIRE
#           SI p a une place libre ALORS ajouter (s, p)
#           SINON SI un admis x de p est moins prioritaire que s ALORS ajouter (s, p)
#   RETOURNER paires
# FIN
# Zéro paire bloquante = appariement stable (aucune jalousie justifiée).
# ------------------------------------------------------------------------------
def blocking_pairs(matching, true_prefs, program_priorities, capacities):
    assign = assignment_of(matching)
    pairs = []
    for s in true_prefs:
        prefs = true_prefs[s]
        if s in assign:
            limit = prefs.index(assign[s])
        else:
            limit = len(prefs)

        for k in range(limit):                          # programmes préférés à l'affectation
            p = prefs[k]
            admitted = matching[p]
            if len(admitted) < capacities[p]:
                pairs.append((s, p))
            else:
                rank_s = program_priorities[p].index(s)
                for x in admitted:
                    if rank_s < program_priorities[p].index(x):
                        pairs.append((s, p))
                        break
    return pairs


def is_feasible(matching, students, capacities):
    """Vérifie : capacités respectées, personne admis deux fois, aucun inconnu."""
    seen = []
    for p in matching:
        if len(matching[p]) > capacities[p]:
            return False
        for s in matching[p]:
            if s in seen or s not in students:
                return False
            seen.append(s)
    return True
