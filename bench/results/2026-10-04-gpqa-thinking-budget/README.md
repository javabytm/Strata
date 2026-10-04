# 2026-10-04: what a thinking budget costs on GPQA Diamond (IQ3_S)

Question: `reasoning_budget_tokens` caps the thinking at N tokens and then makes the model answer
(`docs/DETAILS.md`). It is a cap, not a recomputation - the engine carries on from what it already holds - so it does
not cost throughput. But does it cost accuracy? On a reasoning benchmark where the thinking runs to tens of
thousands of tokens, the answer decides whether the setting is safe to turn on.

Method: GPQA Diamond, lm-eval's `cot_zeroshot` prompt and both filters, greedy decode, against the local server.
Qwen3.8-Flash-Next IQ3_S, RTX 5080, 262144 context, vision on, no `reasoning_budget_tokens` set. The runner retries
once at 30,000 tokens when the first 12,000-token attempt returns no text, so the completion count below is both
attempts summed and is an upper bound on the thinking a question used. **92 of the 198 questions** were answered
when this was written; the run was still going.

| Thinking used by | n | min | median | p90 | max |
|---|---:|---:|---:|---:|---:|
| the **71 correct** answers | 71 | 541 | 4,814 | 27,542 | 40,389 |
| the **12 wrong** answers | 12 | 519 | 20,961 | 41,461 | 41,619 |

If the thinking were capped at N, and an answer that needed more than N is counted as lost:

| Budget (tokens) | Correct answers cut | Accuracy over the 92 |
|---:|---:|---:|
| 4,000 | 37 of 71 | 37.0% |
| 8,000 | 24 of 71 | 51.1% |
| 12,000 | 15 of 71 | 60.9% |
| 16,000 | 15 of 71 | 60.9% |
| 24,000 | 14 of 71 | 62.0% |
| 32,000 | 6 of 71 | 70.7% |
| **none (measured)** | **0** | **77.2%** |

Reading:
- **This model needs its long thinking.** The correct answers have a median of 4,814 thinking tokens but a p90 of
  27,542: more than half of them are past 4,000, and a fifth are past 27,000. Every budget short enough to save real
  time cuts correct answers faster than it saves anything - 12,000 tokens costs 16 points.
- **A budget does not rescue the questions that stall.** 9 of the 92 (13%) produced no text at all, each burning the
  full 12,000 + 30,000, 99 minutes in total. Re-run, 9 of 10 such questions stalled again or answered wrong, so they
  fail whether or not the thinking is capped - the cap only makes them fail faster.
- The "cut" column is the pessimistic reading: it assumes an answer that used more than N would have come out wrong
  at N. A forced wrap-up can guess right, so the real cost may be a little smaller - but the loss of correct answers
  is the dominant term, not the gain.
- Reading it the other way: the setting is best left off for reasoning work and turned on per request where a quick
  answer matters more than a deep one, which is what an opt-in setting is for.

Two things this does not measure yet: the score of the full 198 (the run was at 92), and whether a wrap-up line
produces a *different* answer rather than merely a shorter one. Both need the run to finish.
