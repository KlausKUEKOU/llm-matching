# Un modèle de langage peut-il résoudre un problème d'appariement ?
#
#   Méthode A : le LLM lit chaque lettre et donne un classement des programmes,
#               puis l'acceptation différée (DA) fait l'appariement.
#   Méthode B : le LLM lit tout et donne directement l'affectation.
#
# Points de comparaison : DA sur les vraies préférences (oracle) et DA sur un
# classement obtenu par mots-clés (TF-IDF).
#
# Lancement :
#   python run_experiment.py          utilise les réponses du LLM en cache (data/llm_cache/)
#   python run_experiment.py --live   interroge le modèle configuré dans src/llm.py

import json
import math
import random
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.generate import PROGRAMS, program_priorities
from src.matching import assignment_of, blocking_pairs, deferred_acceptance, is_feasible
from src import prompts

ROOT = Path(__file__).parent
LIVE = "--live" in sys.argv


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            cells = []
            for x in row:
                if isinstance(x, float):
                    cells.append("%.3f" % x)
                else:
                    cells.append(str(x))
            f.write(",".join(cells) + "\n")


def mean(values):
    return sum(values) / len(values)


# ------------------------------------------------------------------ 1. données
truth = read_json(ROOT / "data/students_ground_truth.json")
apps = read_json(ROOT / "data/applications.json")
codes = list(PROGRAMS)

capacities = {}
for c in codes:
    capacities[c] = PROGRAMS[c]["capacity"]

true_prefs = {}
utilities = {}
for s in truth:
    true_prefs[s["id"]] = s["true_ranking"]
    utilities[s["id"]] = s["utilities"]
students = list(true_prefs)
priorities = program_priorities(truth)

# ------------------------------------------------------------------ 2. réponses du LLM
if LIVE:
    from src.llm import complete, parse_json
    llm_prefs = {}
    for a in apps:
        question = prompts.ELICITATION.format(programs=prompts.describe_programs(PROGRAMS),
                                              n=len(codes), letter=a["letter"])
        llm_prefs[a["id"]] = parse_json(complete(question))["ranking"]
    listing = ""
    for a in apps:
        listing = listing + "- " + a["id"] + ": " + a["letter"] + " Grades: " + json.dumps(a["grades"]) + "\n"
    question = prompts.DIRECT_ASSIGNMENT.format(k=len(codes), programs=prompts.describe_programs(PROGRAMS, True),
                                                applications=listing)
    llm_direct = parse_json(complete(question, max_tokens=800))["assignment"]
else:
    llm_prefs = read_json(ROOT / "data/llm_cache/elicitation_claude-opus-5-5.json")["rankings"]
    llm_direct = read_json(ROOT / "data/llm_cache/direct_assignment_claude-opus-5-5.json")["assignment"]


# ------------------------------------------------------------------ 3. référence naïve : TF-IDF
# ALGORITHME ClassementParMotsCles
#   découper chaque texte en mots (minuscules, au moins 2 lettres, sans mots vides)
#   idf[mot] <- ln((1 + N) / (1 + nb de textes contenant le mot)) + 1
#   vecteur(texte)[mot] <- nb d'occurrences x idf[mot], puis normalisé à la longueur 1
#   score(lettre, programme) <- produit scalaire des deux vecteurs (similarité cosinus)
#   classement <- programmes triés par score décroissant
def tokenize(text):
    words = re.findall(r"(?u)\b\w\w+\b", text.lower())
    kept = []
    for w in words:
        if w not in ENGLISH_STOP_WORDS:
            kept.append(w)
    return kept


def tfidf_vector(words, idf):
    vec = {}
    for w in words:
        if w in idf:
            vec[w] = vec.get(w, 0) + idf[w]
    squares = 0.0
    for w in vec:
        squares = squares + vec[w] * vec[w]
    norm = math.sqrt(squares)
    if norm > 0:
        for w in vec:
            vec[w] = vec[w] / norm
    return vec


def dot(u, v):
    total = 0.0
    for w in u:
        if w in v:
            total = total + u[w] * v[w]
    return total


program_texts = []
for c in codes:
    program_texts.append(PROGRAMS[c]["name"] + " " + PROGRAMS[c]["description"])
all_texts = list(program_texts)
for a in apps:
    all_texts.append(a["letter"])

doc_count = {}
for text in all_texts:
    for w in set(tokenize(text)):
        doc_count[w] = doc_count.get(w, 0) + 1
N = len(all_texts)
idf = {}
for w in doc_count:
    idf[w] = math.log((1 + N) / (1 + doc_count[w])) + 1

program_vectors = []
for text in program_texts:
    program_vectors.append(tfidf_vector(tokenize(text), idf))

kw_prefs = {}
for a in apps:
    letter_vec = tfidf_vector(tokenize(a["letter"]), idf)
    scored = []
    for k in range(len(codes)):
        scored.append((-dot(letter_vec, program_vectors[k]), k))    # égalité : ordre des programmes
    scored.sort()
    ranking = []
    for pair in scored:
        ranking.append(codes[pair[1]])
    kw_prefs[a["id"]] = ranking


# ------------------------------------------------------------------ 4. qualité de la lecture des lettres
# Tau de Kendall entre deux classements de 4 éléments :
#   pour chaque paire de programmes, +1 si les deux classements sont d'accord, -1 sinon,
#   puis on divise par le nombre de paires (6).
def kendall_tau(ranking_a, ranking_b):
    concordant = 0
    discordant = 0
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            x, y = codes[i], codes[j]
            sign_a = ranking_a.index(x) - ranking_a.index(y)
            sign_b = ranking_b.index(x) - ranking_b.index(y)
            if sign_a * sign_b > 0:
                concordant = concordant + 1
            else:
                discordant = discordant + 1
    return (concordant - discordant) / (concordant + discordant)


elicitation_rows = []
for name, prefs in [("LLM (Claude)", llm_prefs), ("TF-IDF keywords", kw_prefs)]:
    top_ok, last_ok, exact, taus = [], [], [], []
    for s in students:
        top_ok.append(1 if prefs[s][0] == true_prefs[s][0] else 0)
        last_ok.append(1 if prefs[s][-1] == true_prefs[s][-1] else 0)
        exact.append(1 if prefs[s] == true_prefs[s] else 0)
        taus.append(kendall_tau(prefs[s], true_prefs[s]))
    elicitation_rows.append([name, mean(top_ok), mean(last_ok), mean(exact), mean(taus)])
write_csv(ROOT / "results/elicitation_accuracy.csv",
          ["Elicitation method", "Top choice correct", "Last choice correct", "Full ranking exact",
           "Mean Kendall tau"], elicitation_rows)

# ------------------------------------------------------------------ 5. les quatre appariements
direct = {}
for c in codes:
    direct[c] = list(llm_direct.get(c, []))

methods = [
    ("Oracle DA (true preferences)", deferred_acceptance(true_prefs, priorities, capacities)),
    ("LLM elicitation + DA", deferred_acceptance(llm_prefs, priorities, capacities)),
    ("Keyword elicitation + DA", deferred_acceptance(kw_prefs, priorities, capacities)),
    ("LLM direct assignment", direct),
]
oracle = assignment_of(methods[0][1])

result_rows = []
assignment_rows = {}
for name, matching in methods:
    assign = assignment_of(matching)
    ranks, first, util_total, worse = [], 0, 0.0, 0
    for s in students:
        if s in assign:
            r = true_prefs[s].index(assign[s]) + 1
            ranks.append(r)
            if r == 1:
                first = first + 1
            util_total = util_total + utilities[s][assign[s]]
            if true_prefs[s].index(assign[s]) > true_prefs[s].index(oracle[s]):
                worse = worse + 1
        assignment_rows.setdefault(s, []).append(assign.get(s, ""))
    bp = blocking_pairs(matching, true_prefs, priorities, capacities)
    students_in_bp = []
    for pair in bp:
        if pair[0] not in students_in_bp:
            students_in_bp.append(pair[0])
    result_rows.append([name, is_feasible(matching, students, capacities), len(students) - len(assign),
                        first / len(students), mean(ranks), util_total, len(bp), len(students_in_bp), worse])
write_csv(ROOT / "results/matching_comparison.csv",
          ["Method", "Feasible", "Unassigned", "Got true 1st choice", "Mean true rank of assignment",
           "Total utility", "Blocking pairs (true prefs)", "Students in a blocking pair",
           "Students worse off than oracle"], result_rows)
rows = []
for s in students:
    rows.append([s] + assignment_rows[s])
header = ["student"]
for m in methods:
    header.append(m[0])
write_csv(ROOT / "results/assignments.csv", header, rows)

# ------------------------------------------------------------------ 6. simulation : erreurs de lecture
# ALGORITHME PropagationDesErreurs
# POUR p allant de 0 à 0,6 par pas de 0,05 FAIRE
#     REPETER 300 fois
#         pour chaque étudiant, partir de son vrai classement et échanger chaque paire de
#         programmes voisins avec la probabilité p
#         calculer DA sur ces classements bruités
#         compter les paires bloquantes (vraies préférences) et les 1ers choix obtenus
#     faire la moyenne des 300 essais
rng = random.Random(0)
noise_rows = []
for step in range(13):
    p = step * 0.05
    total_bp, total_first, total_top = 0.0, 0.0, 0.0
    for rep in range(300):
        noisy = {}
        for s in students:
            ranking = list(true_prefs[s])
            for k in range(len(ranking) - 1):
                if rng.random() < p:
                    ranking[k], ranking[k + 1] = ranking[k + 1], ranking[k]
            noisy[s] = ranking
        matching = deferred_acceptance(noisy, priorities, capacities)
        assign = assignment_of(matching)
        total_bp = total_bp + len(blocking_pairs(matching, true_prefs, priorities, capacities))
        for s in students:
            if assign[s] == true_prefs[s][0]:
                total_first = total_first + 1
            if noisy[s][0] == true_prefs[s][0]:
                total_top = total_top + 1
    noise_rows.append([p, total_bp / 300, total_first / (300 * len(students)), total_top / (300 * len(students))])
write_csv(ROOT / "results/elicitation_noise_simulation.csv",
          ["swap_prob", "blocking_pairs", "first_choice", "exact_top"], noise_rows)

# ------------------------------------------------------------------ 7. graphiques
names, first_pct, n_blocking = [], [], []
for r in reversed(result_rows):                     # ordre inversé : la 1re méthode en haut
    names.append(r[0])
    first_pct.append(100 * r[3])
    n_blocking.append(r[6])
colors = ["#d62728", "#ff7f0e", "#1f77b4", "#7f7f7f"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].barh(names, first_pct, color=colors)
axes[0].set_xlabel("Students who get their true first choice (%)")
axes[1].barh(names, n_blocking, color=colors)
axes[1].set_xlabel("Blocking pairs (0 = stable)")
axes[1].set_yticks([])
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.suptitle("Matching 24 students to 4 Master programmes: who gets what, and is it stable?")
fig.tight_layout()
fig.savefig(ROOT / "figures/matching_comparison.png", dpi=130)
plt.close(fig)

probs, mean_bp, mean_first = [], [], []
for r in noise_rows:
    probs.append(r[0])
    mean_bp.append(r[1])
    mean_first.append(100 * r[2])
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(probs, mean_bp, "o-", color="#c0392b", label="Blocking pairs (left)")
ax1.set_xlabel("Probability of swapping two adjacent programmes in an elicited ranking")
ax1.set_ylabel("Blocking pairs w.r.t. true preferences")
ax2 = ax1.twinx()
ax2.plot(probs, mean_first, "s-", color="#2471a3", label="Get true 1st choice (right)")
ax2.set_ylabel("%")
lines = ax1.get_lines() + ax2.get_lines()
labels = []
for line in lines:
    labels.append(line.get_label())
ax1.legend(lines, labels, fontsize=8, loc="center right")
ax1.set_title("DA is stable only with respect to the preferences it is given")
fig.tight_layout()
fig.savefig(ROOT / "figures/elicitation_noise.png", dpi=130)
plt.close(fig)

for row in elicitation_rows + result_rows:
    print(row)
