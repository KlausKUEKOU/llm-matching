# Solving a Matching Problem with a Language Model

Admissions to Master programmes are a matching problem: students have preferences over
programmes, programmes rank students, and seats are limited. The deferred acceptance algorithm of
Gale and Shapley (1962) gives a stable matching: no student can claim a seat that was given to
someone with a lower priority. But the algorithm needs rankings, while real students write
motivation letters.

This project compares two ways of using a language model (LLM):

- **A.** The LLM reads each letter and returns a ranking of the programmes. Deferred acceptance then
  computes the matching.
- **B.** The LLM reads all the letters and grades and returns the assignment directly.

## Data

The market is synthetic so that the true preferences are known (`src/generate.py`):

- 24 students, 4 programmes (Econometrics & Data Science, Quantitative Finance, Development
  Economics, AI & Decision Science), 6 seats each.
- Each student has hidden utilities. The letter only gives indirect clues: "I follow the BRVM every
  day" points to finance, "my mother's cassava business" to development, "I coded a chess engine"
  to AI. Programme names never appear and the order of the sentences is random.
- Each programme ranks students with a weighted average of the grades it cares about.

Four methods are compared with the true preferences: deferred acceptance on the true rankings
(oracle), on the LLM rankings, on rankings from keyword matching (TF-IDF), and the direct LLM
assignment.

## Algorithm

```
DEFERRED ACCEPTANCE
    every student is free ; every programme holds nobody
    WHILE a free student s still has programmes to ask
        p <- best programme that s has not asked yet
        p temporarily holds s
        IF p holds more students than its capacity THEN
            p rejects its lowest-priority student, who becomes free again
    END WHILE
    RETURN the students held by each programme

BLOCKING PAIR
    (s, p) is a blocking pair if s prefers p to her assignment
    and p has a free seat or holds a student with lower priority than s.
    No blocking pair = stable matching.
```

## Results

### Reading preferences from the letters

| Method | Top choice correct | Last choice correct | Full ranking exact | Kendall tau |
|---|---|---|---|---|
| LLM (Claude) | 100% | 100% | 100% | 1.00 |
| TF-IDF keywords | 33% | 13% | 4% | 0.10 |

Keyword matching fails because the clues use different words from the programme descriptions. The
LLM understands the meaning of the sentences.

Important: the cached answers come from the same model that wrote the letter generator, so 100% is
an upper bound. For an independent test, run `--live` with another model (for example Qwen with
Ollama).

### The matchings

| Method | Got true 1st choice | Mean rank obtained | Total utility | Blocking pairs | Worse off than oracle |
|---|---|---|---|---|---|
| Oracle DA (true preferences) | 83% | 1.21 | 29.7 | 0 | 0 |
| LLM rankings + DA | 83% | 1.21 | 29.7 | 0 | 0 |
| Keyword rankings + DA | 29% | 2.46 | 4.5 | 15 | 16 |
| LLM direct assignment | 88% | 1.13 | 34.0 | 1 | 0 |

![Comparison](figures/matching_comparison.png)

- LLM rankings + DA give exactly the oracle matching: stable, and the best stable matching for the
  students.
- The direct LLM assignment gives more first choices but is not stable. It put S09 in AI & Decision
  Science (her first choice) and S03 in Development (her second choice). S14 also wanted AI &
  Decision Science and has a higher priority than S09, so S14 could contest the decision. The model
  traded stability for total satisfaction without saying which rule it followed.
- With keyword rankings, DA is stable for the rankings it receives, but there are 15 blocking pairs
  with respect to the true preferences, and 16 students out of 24 are worse off.

### Effect of reading errors

Simulation: the true rankings are perturbed by swapping two neighbouring programmes with
probability p, then DA is run again (300 draws for each p).

![Noise](figures/elicitation_noise.png)

Each additional 5 points of error probability add about one blocking pair. With 20% of errors, 71%
of students get their first choice (83% without errors) and there are about 4 blocking pairs. The
quality of the matching depends mostly on the quality of the reading step.

## Conclusion

- Use the LLM to read the texts, and keep an algorithm with known properties (stability,
  strategy-proofness) to allocate the seats.
- A direct LLM allocation can look better on average but it breaks priorities in a way that is
  hard to check and to explain to the students who lose.
- The rankings extracted by the LLM should be checked, for example by asking each student to
  confirm them.

## Run it

```bash
pip install -r requirements.txt
python -m src.generate          # rebuild the market (fixed seed)
python run_experiment.py        # uses the cached LLM answers, no API key needed

# with a live model
export LLM_PROVIDER=anthropic LLM_MODEL=claude-opus-5-5 ANTHROPIC_API_KEY=...
# or: export LLM_PROVIDER=openai OPENAI_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen3:8b
python run_experiment.py --live
```

```
src/generate.py      synthetic market: utilities, letters, grades, priorities
src/matching.py      deferred acceptance, blocking pairs, feasibility check
src/prompts.py       prompts sent to the model
src/llm.py           client for Anthropic or any OpenAI-compatible server
data/llm_cache/      cached model answers (Claude Opus 5.5)
results/, figures/   outputs
```

Author: Klaus KUEKOU
