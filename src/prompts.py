"""Prompts sent to the language model."""
import json

ELICITATION = """You help a university admissions office. Read the application letter of one student
and infer how the student ranks the following Master programmes, from most to least preferred.

Programmes:
{programs}

Rules:
- Base the ranking only on what the letter says (interests, experiences, career plans, dislikes).
- Programmes are not named in letters: infer them from indirect cues.
- Give a strict ranking of all {n} programme codes.

Letter:
\"\"\"{letter}\"\"\"

Answer with JSON only: {{"ranking": ["CODE", "CODE", ...]}}"""

DIRECT_ASSIGNMENT = """You are the admissions committee of a school with {k} Master programmes.
Assign every student to exactly one programme. Respect the capacities. Try to give students
programmes they want, and give contested seats to the students with the best relevant grades.

Programmes (code, description, capacity, grades that matter):
{programs}

Applications (id, letter, grades out of 20):
{applications}

Answer with JSON only: {{"assignment": {{"CODE": ["S01", ...], ...}}}}"""


def describe_programs(programs, with_capacity=False):
    lines = []
    for code, p in programs.items():
        line = f"- {code}: {p['name']}. {p['description']}"
        if with_capacity:
            line += f" Capacity {p['capacity']}. Grades: {json.dumps(p['grade_weights'])}"
        lines.append(line)
    return "\n".join(lines)
