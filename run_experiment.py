"""Can a language model solve a matching problem? Two ways to use it, compared with ground truth.

  A. LLM as *preference elicitor*: it reads each letter and outputs a ranking; the matching is then
     computed by deferred acceptance (DA), which guarantees stability.
  B. LLM as *matchmaker*: it reads everything and outputs the assignment directly.

Benchmarks: oracle DA on the true preferences, and DA on rankings from a TF-IDF keyword matcher.

Usage:
  python run_experiment.py            uses cached LLM answers in data/llm_cache/
  python run_experiment.py --live     queries the model configured in src/llm.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.generate import PROGRAMS, program_priorities
from src.matching import assignment_of, blocking_pairs, deferred_acceptance, is_feasible
from src import prompts

ROOT = Path(__file__).parent
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
parser = argparse.ArgumentParser()
parser.add_argument("--live", action="store_true")
args = parser.parse_args()

truth = json.load(open(ROOT / "data/students_ground_truth.json", encoding="utf-8"))
apps = json.load(open(ROOT / "data/applications.json", encoding="utf-8"))
codes = list(PROGRAMS)
caps = {c: PROGRAMS[c]["capacity"] for c in codes}
true_prefs = {s["id"]: s["true_ranking"] for s in truth}
util = {s["id"]: s["utilities"] for s in truth}
prio = program_priorities(truth)
students = list(true_prefs)

# ------------------------------------------------------------------ preference elicitation
if args.live:
    from src.llm import complete, parse_json
    model_tag = "live"
    llm_prefs = {}
    for a in apps:
        p = prompts.ELICITATION.format(programs=prompts.describe_programs(PROGRAMS), n=len(codes), letter=a["letter"])
        llm_prefs[a["id"]] = parse_json(complete(p))["ranking"]
    listing = "\n".join(f"- {a['id']}: {a['letter']} Grades: {json.dumps(a['grades'])}" for a in apps)
    p = prompts.DIRECT_ASSIGNMENT.format(k=len(codes), programs=prompts.describe_programs(PROGRAMS, True),
                                         applications=listing)
    llm_direct = parse_json(complete(p, max_tokens=800))["assignment"]
else:
    cache = ROOT / "data/llm_cache"
    llm_prefs = json.load(open(cache / "elicitation_claude-opus-5-5.json"))["rankings"]
    llm_direct = json.load(open(cache / "direct_assignment_claude-opus-5-5.json"))["assignment"]

# keyword baseline: cosine similarity between the letter and each programme description
docs = [f"{PROGRAMS[c]['name']} {PROGRAMS[c]['description']}" for c in codes]
vec = TfidfVectorizer(stop_words="english").fit(docs + [a["letter"] for a in apps])
sim = cosine_similarity(vec.transform([a["letter"] for a in apps]), vec.transform(docs))
kw_prefs = {a["id"]: [codes[k] for k in np.argsort(-sim[i], kind="stable")] for i, a in enumerate(apps)}

elic_rows = []
for name, prefs in [("LLM (Claude)", llm_prefs), ("TF-IDF keywords", kw_prefs)]:
    taus = [kendalltau([prefs[s].index(c) for c in codes], [true_prefs[s].index(c) for c in codes])[0]
            for s in students]
    elic_rows.append({"Elicitation method": name,
                      "Top choice correct": np.mean([prefs[s][0] == true_prefs[s][0] for s in students]),
                      "Last choice correct": np.mean([prefs[s][-1] == true_prefs[s][-1] for s in students]),
                      "Full ranking exact": np.mean([prefs[s] == true_prefs[s] for s in students]),
                      "Mean Kendall tau": np.mean(taus)})
elic = pd.DataFrame(elic_rows).set_index("Elicitation method")
elic.to_csv(ROOT / "results/elicitation_accuracy.csv", float_format="%.3f")

# ------------------------------------------------------------------ matchings
matchings = {
    "Oracle DA (true preferences)": deferred_acceptance(true_prefs, prio, caps),
    "LLM elicitation + DA": deferred_acceptance(llm_prefs, prio, caps),
    "Keyword elicitation + DA": deferred_acceptance(kw_prefs, prio, caps),
    "LLM direct assignment": {c: list(llm_direct.get(c, [])) for c in codes},
}
oracle = assignment_of(matchings["Oracle DA (true preferences)"])
rows, assignments = [], {}
for name, m in matchings.items():
    a = assignment_of(m)
    assignments[name] = a
    ranks = [true_prefs[s].index(a[s]) + 1 if s in a else np.nan for s in students]
    bp = blocking_pairs(m, true_prefs, prio, caps)
    rows.append({
        "Method": name, "Feasible": is_feasible(m, students, caps),
        "Unassigned": sum(s not in a for s in students),
        "Got true 1st choice": np.mean([r == 1 for r in ranks]),
        "Mean true rank of assignment": np.nanmean(ranks),
        "Total utility": sum(util[s][a[s]] for s in students if s in a),
        "Blocking pairs (true prefs)": len(bp),
        "Students in a blocking pair": len({s for s, _ in bp}),
        "Students worse off than oracle": sum(true_prefs[s].index(a[s]) > true_prefs[s].index(oracle[s])
                                              for s in students if s in a),
    })
res = pd.DataFrame(rows).set_index("Method")
res.to_csv(ROOT / "results/matching_comparison.csv", float_format="%.3f")
pd.DataFrame(assignments).rename_axis("student").to_csv(ROOT / "results/assignments.csv")

# ------------------------------------------------------------------ robustness: how do elicitation errors propagate?
rng = np.random.default_rng(0)
noise_rows = []
for p in np.linspace(0, 0.6, 13):
    for rep in range(300):
        noisy = {}
        for s, pr in true_prefs.items():
            pr = pr.copy()
            for k in range(len(pr) - 1):          # each adjacent pair swapped with probability p
                if rng.random() < p:
                    pr[k], pr[k + 1] = pr[k + 1], pr[k]
            noisy[s] = pr
        m = deferred_acceptance(noisy, prio, caps)
        a = assignment_of(m)
        noise_rows.append({"swap_prob": p, "blocking_pairs": len(blocking_pairs(m, true_prefs, prio, caps)),
                           "first_choice": np.mean([a[s] == true_prefs[s][0] for s in students]),
                           "exact_top": np.mean([noisy[s][0] == true_prefs[s][0] for s in students])})
noise = pd.DataFrame(noise_rows).groupby("swap_prob").mean()
noise.to_csv(ROOT / "results/elicitation_noise_simulation.csv", float_format="%.3f")

# ------------------------------------------------------------------ figures
colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#d62728"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].barh(res.index[::-1], res["Got true 1st choice"][::-1] * 100, color=colors[::-1])
axes[0].set_xlabel("Students who get their true first choice (%)")
axes[1].barh(res.index[::-1], res["Blocking pairs (true prefs)"][::-1], color=colors[::-1])
axes[1].set_xlabel("Blocking pairs (0 = stable)")
axes[1].set_yticks([])
fig.suptitle("Matching 24 students to 4 Master programmes: who gets what, and is it stable?")
fig.tight_layout()
fig.savefig(ROOT / "figures/matching_comparison.png")
plt.close(fig)

fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(noise.index, noise["blocking_pairs"], "o-", color="#c0392b", label="Blocking pairs (left)")
ax1.set_xlabel("Probability of swapping two adjacent programmes in an elicited ranking")
ax1.set_ylabel("Blocking pairs w.r.t. true preferences")
ax2 = ax1.twinx()
ax2.plot(noise.index, noise["first_choice"] * 100, "s-", color="#2471a3", label="Get true 1st choice (right)")
ax2.set_ylabel("%")
ax2.spines["top"].set_visible(False)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
ax1.set_title("DA is stable only with respect to the preferences it is given")
fig.tight_layout()
fig.savefig(ROOT / "figures/elicitation_noise.png")
plt.close(fig)

pd.set_option("display.width", 220)
print(elic.round(3)); print(); print(res.round(3)); print(); print(noise.round(3))
