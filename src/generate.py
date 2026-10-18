# Marché synthétique d'admission en Master, avec une vérité connue.
#
# Chaque étudiant a des utilités cachées pour les 4 programmes. Sa lettre de motivation
# est fabriquée à partir de ces utilités, avec des indices indirects (stages, cours aimés,
# projets de carrière) : le nom des programmes n'apparaît jamais.
# Les programmes classent les étudiants selon une moyenne pondérée des notes utiles.

import json
import random
from pathlib import Path

import numpy as np

PROGRAMS = {
    "ECO_DS": {"name": "Econometrics & Data Science", "capacity": 6,
               "description": "Causal inference, panel data, machine learning for economic data, large surveys.",
               "grade_weights": {"econometrics": 0.5, "statistics": 0.3, "programming": 0.2}},
    "QFIN": {"name": "Quantitative Finance", "capacity": 6,
             "description": "Asset pricing, derivatives, risk management, financial time series, trading.",
             "grade_weights": {"mathematics": 0.5, "statistics": 0.3, "economics": 0.2}},
    "DEV": {"name": "Development Economics", "capacity": 6,
            "description": "Poverty, agriculture, public policy evaluation, field experiments in Africa.",
            "grade_weights": {"economics": 0.5, "econometrics": 0.3, "statistics": 0.2}},
    "AIDS": {"name": "AI & Decision Science", "capacity": 6,
             "description": "Optimisation, game theory, social choice, reinforcement learning, algorithms.",
             "grade_weights": {"mathematics": 0.4, "programming": 0.4, "statistics": 0.2}},
}
SUBJECTS = ["mathematics", "statistics", "econometrics", "economics", "programming"]

# Indices forts : typiques du 1er choix. Indices faibles : intérêt modéré. Rejet : dernier choix.
STRONG = {
    "ECO_DS": ["My bachelor thesis estimated the returns to schooling with instrumental variables, and I loved "
               "every step of cleaning the census microdata.",
               "I want to spend my career finding out what really causes what in messy administrative datasets.",
               "During my internship at the national statistics institute I built a gradient-boosting model to "
               "impute missing household incomes."],
    "QFIN": ["I have been pricing options on a spreadsheet since my second year and I follow the BRVM every day.",
             "My dream job is on a trading desk or in the risk department of a large bank.",
             "My internship at a brokerage firm in Abidjan convinced me that volatility is the most fascinating "
             "object in economics."],
    "DEV": ["Growing up in a farming village, I saw how a small loan changed my mother's cassava business.",
            "I want to design and evaluate programmes that reduce rural poverty, ideally with randomised trials.",
            "I volunteered for a year with an NGO measuring the impact of school meals in northern Cameroon."],
    "AIDS": ["I spend my evenings on competitive programming and I recently coded a small chess engine.",
             "Voting rules and fair division problems are what got me into economics in the first place.",
             "I would like to build recommendation and allocation algorithms that are both efficient and fair."],
}
WEAK = {
    "ECO_DS": "Regression analysis is a useful tool and I would not mind deepening it.",
    "QFIN": "Markets interest me somewhat, mainly as a way to understand savings.",
    "DEV": "Public policy questions in Africa matter to me, even if they are not my main focus.",
    "AIDS": "I find algorithms elegant, although I am not a strong coder yet.",
}
AVERSE = {
    "ECO_DS": "Spending months cleaning survey files is honestly not what I am looking for.",
    "QFIN": "I am not attracted by the world of finance and trading floors.",
    "DEV": "I prefer theoretical or market questions to fieldwork and policy evaluation.",
    "AIDS": "Heavy programming and abstract algorithms are not where I see myself.",
}
OPENERS = ["I hold a bachelor's degree in economics from the University of Yaoundé II.",
           "I graduated in applied mathematics and economics from the University of Dschang.",
           "After a licence in economics at Université Félix Houphouët-Boigny, I worked for one year.",
           "I completed my undergraduate studies in statistics at ISSEA.",
           "I studied economics and management at the University of Douala."]


def rank_programs(codes, utilities):
    """Renvoie les codes triés de l'utilité la plus haute à la plus basse (tri par sélection)."""
    remaining = list(range(len(codes)))
    order = []
    while len(remaining) > 0:
        best = remaining[0]
        for k in remaining:
            if utilities[k] > utilities[best]:
                best = k
        order.append(best)
        remaining.remove(best)
    return order


# ------------------------------------------------------------------------------
# ALGORITHME FabriquerMarche
# POUR chaque étudiant FAIRE
#     tirer 4 utilités au hasard (loi normale), bonus de 1,5 à un programme tiré au sort
#     classement_vrai <- programmes triés par utilité décroissante
#     tirer 5 notes entre 8 et 20
#     lettre <- phrase d'ouverture
#              + 2 indices forts du 1er choix
#              + 1 indice fort du 2e choix si son utilité > 0,6, sinon 1 indice faible
#              + 1 indice faible du 3e choix (une fois sur deux)
#              + 1 phrase de rejet du dernier choix
#     mélanger les phrases (l'ordre ne trahit pas le classement)
# FIN POUR
# ------------------------------------------------------------------------------
def make_market(n_students=24, seed=7):
    rng = np.random.default_rng(seed)
    pyrand = random.Random(seed)
    codes = list(PROGRAMS)
    students = []

    for s in range(n_students):
        util = rng.normal(0, 1, len(codes))
        favourite = rng.integers(len(codes))
        util[favourite] = util[favourite] + 1.5

        order_idx = rank_programs(codes, util)
        order = []
        for k in order_idx:
            order.append(codes[k])

        grades = {}
        for subject in SUBJECTS:
            g = rng.normal(13, 2.2)
            if g < 8:
                g = 8
            if g > 19:
                g = 19
            grades[subject] = float(g)

        opener = pyrand.choice(OPENERS)
        body = pyrand.sample(STRONG[order[0]], 2)
        if util[order_idx[1]] > 0.6:
            body.append(pyrand.choice(STRONG[order[1]]))
        else:
            body.append(WEAK[order[1]])
        if pyrand.random() < 0.5:
            body.append(WEAK[order[2]])
        body.append(AVERSE[order[3]])
        pyrand.shuffle(body)

        letter = opener
        for sentence in body:
            letter = letter + " " + sentence

        rounded_grades = {}
        utilities = {}
        for subject in grades:
            rounded_grades[subject] = round(grades[subject], 1)
        for k in range(len(codes)):
            utilities[codes[k]] = round(float(util[k]), 3)

        students.append({"id": "S%02d" % (s + 1), "letter": letter, "grades": rounded_grades,
                         "true_ranking": order, "utilities": utilities})
    return students


# ------------------------------------------------------------------------------
# ALGORITHME PrioritesDesProgrammes
# POUR chaque programme p FAIRE
#     POUR chaque étudiant s : score[s] <- somme des (poids x note) des matières utiles à p
#     priorites[p] <- étudiants triés par score décroissant (égalité : ordre des identifiants)
# ------------------------------------------------------------------------------
def program_priorities(students):
    prio = {}
    for code in PROGRAMS:
        weights = PROGRAMS[code]["grade_weights"]
        scored = []
        for s in students:
            score = 0
            for subject in weights:
                score = score + weights[subject] * s["grades"][subject]
            scored.append((-score, s["id"]))            # signe moins : tri décroissant
        scored.sort()
        prio[code] = []
        for pair in scored:
            prio[code].append(pair[1])
    return prio


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    students = make_market()
    (root / "data").mkdir(exist_ok=True)

    with open(root / "data/students_ground_truth.json", "w") as f:
        json.dump(students, f, indent=1, ensure_ascii=False)

    public = []
    for s in students:
        public.append({"id": s["id"], "letter": s["letter"], "grades": s["grades"]})
    with open(root / "data/applications.json", "w") as f:
        json.dump(public, f, indent=1, ensure_ascii=False)

    with open(root / "data/programs.json", "w") as f:
        json.dump(PROGRAMS, f, indent=1, ensure_ascii=False)
    print("applications écrites :", len(students))
