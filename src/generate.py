"""Synthetic school-choice market with known ground truth.

Each student has latent utilities over the Master programs. Her application letter is generated
from those utilities with *indirect* cues (courses enjoyed, internships, career plans), so that
recovering the ranking requires reading comprehension rather than keyword matching. Programs rank
students by a weighted average of the grades that matter for them.
"""
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

# Indirect cues: each signals one program without naming it. Strong cues are typical of a first
# choice; weak cues of a moderate interest; "aversion" cues of a last choice.
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


def make_market(n_students=24, seed=7):
    rng = np.random.default_rng(seed)
    pyrand = random.Random(seed)
    codes = list(PROGRAMS)
    students = []
    for s in range(n_students):
        util = rng.normal(0, 1, len(codes))
        util[rng.integers(len(codes))] += 1.5           # most students have a clear favourite
        order = [codes[k] for k in np.argsort(-util)]
        grades = {sub: float(np.clip(rng.normal(13, 2.2), 8, 19)) for sub in SUBJECTS}
        sentences = [pyrand.choice(OPENERS)]
        sentences += pyrand.sample(STRONG[order[0]], 2)
        sentences.append(pyrand.choice(STRONG[order[1]]) if util[np.argsort(-util)[1]] > 0.6 else WEAK[order[1]])
        sentences.append(WEAK[order[2]] if pyrand.random() < 0.5 else "")
        sentences.append(AVERSE[order[3]])
        body = [x for x in sentences[1:] if x]
        pyrand.shuffle(body)                          # cue order says nothing about preference order
        letter = " ".join([sentences[0]] + body)
        students.append({"id": f"S{s + 1:02d}", "letter": letter, "grades": {k: round(v, 1) for k, v in grades.items()},
                         "true_ranking": order, "utilities": dict(zip(codes, np.round(util, 3).tolist()))})
    return students


def program_priorities(students):
    """Each program ranks students by its weighted grade average (ties broken by id)."""
    prio = {}
    for code, p in PROGRAMS.items():
        score = {s["id"]: sum(w * s["grades"][sub] for sub, w in p["grade_weights"].items()) for s in students}
        prio[code] = sorted(score, key=lambda sid: (-score[sid], sid))
    return prio


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    students = make_market()
    (root / "data").mkdir(exist_ok=True)
    with open(root / "data/students_ground_truth.json", "w") as f:
        json.dump(students, f, indent=1, ensure_ascii=False)
    public = [{"id": s["id"], "letter": s["letter"], "grades": s["grades"]} for s in students]
    with open(root / "data/applications.json", "w") as f:
        json.dump(public, f, indent=1, ensure_ascii=False)
    with open(root / "data/programs.json", "w") as f:
        json.dump(PROGRAMS, f, indent=1, ensure_ascii=False)
    print(f"wrote {len(students)} applications")
