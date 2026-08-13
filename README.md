# NP-Complete Problem Solver Agents

A team project investigating whether Large Language Models can solve NP-Complete problems: benchmarking an off-the-shelf LLM against an exact SMT solver (Z3) across six canonical NPC problems, then improving performance on the hardest problem — Hamiltonian Path — via LoRA fine-tuning (SFT + GRPO) and a ReAct agent equipped with a heuristic repair tool.

**Institution:** University of Tehran
**Course:** Large Language Models
**Authors:** Babak Hosseini Mohtasham, Mahdi Naeni, Mohammad Taha Majlesi
**Full report:** [`NPC-solver-paper.pdf`](./NPC-solver-paper.pdf)
**Additional materials:** [`NPC_SOLVER/Reports/`](./NPC_SOLVER/Reports) *(report and presentation slides)*

*Note: this repository also contains an unrelated `BITCOIN_PRICE_PREDICTOR` project from the same course, not covered by this README because we ended up only presenting the NP project for the course.*

---

## Overview

NP-Complete problems are computationally intractable in the worst case, and exact solvers like Z3 handle them by search rather than pattern recognition. This project asks whether an LLM's learned pattern-recognition ability offers a useful alternative: fast but approximate, in contrast to a solver's exact but potentially slow guarantees. The project benchmarks a state-of-the-art LLM against Z3 on both accuracy and speed across six NP-Complete problems, identifies where the LLM struggles most, and then investigates two ways to close that gap on the hardest case: fine-tuning a smaller open model, and wrapping it in a tool-using agent.

**Objectives:**

1. Generate parameterized, verifiable problem instances for six NP-Complete problems — 3-SAT, Subset Sum (SSP), Minimum Vertex Cover (VC), Maximum Clique, Hamiltonian Path, and Hamiltonian Cycle — each with an automatic solution verifier and a Z3-based exact solver for ground truth.
2. Benchmark Gemini 2.5 Flash zero-shot on 100 instances of each problem, measuring accuracy, F1-score, and solve time against Z3, and analyze which problem parameters (graph size, edge density, clause count, etc.) predict LLM success or failure.
3. Based on that analysis, select the problem where the LLM showed the most room for improvement (Hamiltonian Path) and fine-tune `Llama-3.2-3B-Instruct` with LoRA — first via supervised fine-tuning (SFT), then via GRPO reinforcement learning with two candidate reward functions — and evaluate the fine-tuned models' accuracy on a held-out test set.
4. Build a ReAct agent that uses the fine-tuned model as an initial-solution generator, paired with a 2-opt-style local-search repair tool the agent can invoke to iteratively fix invalid paths, and evaluate the combined system's accuracy and speed against both the base LLM and Z3.

## Methodology

| Stage | Description |
|---|---|
| **Problem generators & verifiers** | Custom parameterized generators for all six problems (`NP_problems.py`), each paired with a solution verifier and a Z3-based exact solver used as ground truth |
| **Baseline evaluation** | Gemini 2.5 Flash prompted zero-shot with structured, parseable-output prompts, over 100 randomly parameterized instances per problem (600 instances total); accuracy, F1, and solve time recorded and compared against Z3 |
| **Difficulty analysis** | A Random Forest classifier trained per problem to predict instance-level LLM success from generator parameters (e.g., graph size, edge density, clause-to-variable ratio), with feature importance used to understand what drives difficulty |
| **Fine-tuning (Hamiltonian Path)** | `unsloth/Llama-3.2-3B-Instruct` fine-tuned with LoRA (rank 64, all linear layers): first SFT on 400 instances to teach output format, then GRPO on 600 instances using one of two reward functions — a simple per-vertex-correctness reward, and a stricter reward based on the longest correct prefix of the proposed path |
| **Agent** | A ReAct agent (built on LangGraph) that takes the fine-tuned model's candidate path, validates it with a path-checking tool, and — if invalid — invokes a 2-opt local-search repair tool (O(C·n²), C=3) to iteratively reduce the number of broken edges in the path |

## Repository Structure

| Path | Description |
|---|---|
| [`NPC-solver-paper.pdf`](./NPC-solver-paper.pdf) | Final written report (top-level copy) |
| [`NPC_SOLVER/Codes/NP_problems.py`](./NPC_SOLVER/Codes/NP_problems.py) | Problem generators, solution verifiers, and visualization for all six NP-Complete problems |
| [`NPC_SOLVER/Codes/NP_solver.py`](./NPC_SOLVER/Codes/NP_solver.py) | Shared evaluation harness: baseline LLM runner, prompt templates per problem, answer extraction, Z3 integration, and result reporting/plotting |
| [`NPC_SOLVER/Codes/solve_NP_problems.ipynb`](./NPC_SOLVER/Codes/solve_NP_problems.ipynb) | Baseline evaluation of Gemini 2.5 Flash against Z3 across all six problems |
| [`NPC_SOLVER/Codes/solve_NP_problems_with_finetuning.ipynb`](./NPC_SOLVER/Codes/solve_NP_problems_with_finetuning.ipynb) | SFT and GRPO fine-tuning of Llama-3.2-3B on Hamiltonian Path, across several dataset/reward configurations, with evaluation |
| [`NPC_SOLVER/Codes/solve_NP_problems_with_agent.ipynb`](./NPC_SOLVER/Codes/solve_NP_problems_with_agent.ipynb), [`..._with_agents_2.ipynb`](./NPC_SOLVER/Codes/solve_NP_problems_with_agents_2.ipynb) | Iterative development of the ReAct agent (multiple prompt/tool design iterations) across 3-SAT, Hamiltonian Path, and Hamiltonian Cycle |
| [`NPC_SOLVER/Codes/Z3Solver.ipynb`](./NPC_SOLVER/Codes/Z3Solver.ipynb) | Standalone Z3 solver reference/testing notebook |
| [`NPC_SOLVER/BaseResults/`](./NPC_SOLVER/BaseResults) | Saved baseline evaluation results (one file per problem) |
| [`NPC_SOLVER/AgentResults/`](./NPC_SOLVER/AgentResults) | Saved final agent evaluation results for Hamiltonian Path |
| [`NPC_SOLVER/Dataset/`](./NPC_SOLVER/Dataset) | Generated problem instance datasets used for fine-tuning and evaluation |
| [`NPC_SOLVER/Models/`](./NPC_SOLVER/Models) | Saved LoRA adapter checkpoint(s) |
| [`NPC_SOLVER/Reports/`](./NPC_SOLVER/Reports) | Report and presentation slides |

## Key Results

**Baseline: Gemini 2.5 Flash vs. Z3, across all six NP-Complete problems (100 instances each):**

| Problem | LLM Accuracy | LLM F1 | LLM Avg. Time | Z3 Avg. Time |
|---|---|---|---|---|
| 3-SAT | 80% | 0.80 | 18.1 s | 0.02 s |
| Subset Sum | 95% | 0.80 | 14.0 s | 0.06 s |
| Vertex Cover | 95% | 0.89 | 22.8 s | 5.9 s |
| Maximum Clique | 75% | 0.71 | 28.5 s | 0.39 s |
| Hamiltonian Path | 100%\* | 1.00\* | 22.3 s | — |
| Hamiltonian Cycle | 95% | 0.67 | 22.8 s | — |

*\*Hamiltonian Path's initial baseline batch scored well on the smaller sampled set used for solver-speed comparison; the larger, more challenging 100-instance benchmark used for the fine-tuning study (below) showed substantially more headroom, which is why this problem was selected for further work.*

**Z3 solved every problem orders of magnitude faster than the LLM** wherever a direct comparison was possible (up to ~900× faster on 3-SAT), confirming that an exact solver remains the better choice when speed and guaranteed correctness both matter — the interesting question this project pursues is whether an LLM-based approach can close the accuracy gap while retaining a speed advantage over cases where Z3 itself becomes slow.

**Fine-tuning alone did not improve Hamiltonian Path performance — it substantially hurt it.** Evaluated on the same challenging 100-instance test set used for the baseline, the SFT-only model solved only 7 of 100 instances, and GRPO training on top of SFT did not recover this gap; both remained far below the base (non-fine-tuned) Gemini model's performance on this harder benchmark. This is a real and informative negative result: a 3B open model fine-tuned with LoRA on a few hundred examples could not match a much larger, general-purpose model's zero-shot reasoning on this task.

**The ReAct agent — using the fine-tuned model as generator plus a 2-opt repair tool — recovered strong performance and generalized to larger graphs.** Despite the fine-tuned model alone performing poorly, wrapping it in the agent (candidate generation → validation → iterative repair) achieved:

| Configuration | Hamiltonian Path Accuracy | Avg. Solve Time |
|---|---|---|
| SFT-only model (no agent) | 7% | — |
| SFT model + ReAct agent | 81.4% | — |
| **GRPO model + ReAct agent** | **82.5%** | **129.6 s** |

The agent's repair tool compensated almost entirely for the fine-tuned generator's weaknesses, successfully solving instances with up to 100 vertices — and while its ~130-second average solve time is slower than the LLM baseline alone, it remains dramatically faster than the hours-long runtimes Z3 requires on the largest Hamiltonian Path instances, illustrating a practical LLM-plus-heuristic middle ground between raw LLM guessing and exact but slow search.

## Reproducing the Results

The notebooks were developed for Google Colab and require API access to external providers.

1. Install dependencies from [`NPC_SOLVER/Codes/requirements.txt`](./NPC_SOLVER/Codes/requirements.txt).
2. Set `GEMINI_API_KEY3` (baseline evaluation and agent notebooks) and `HF_TOKEN` (fine-tuning notebooks) via a `.env` file.
3. Run [`solve_NP_problems.ipynb`](./NPC_SOLVER/Codes/solve_NP_problems.ipynb) first to reproduce the baseline benchmark; results are saved to `NPC_SOLVER/BaseResults/`.
4. Run [`solve_NP_problems_with_finetuning.ipynb`](./NPC_SOLVER/Codes/solve_NP_problems_with_finetuning.ipynb) to reproduce SFT/GRPO training — a GPU runtime is required. Pre-generated datasets are provided in `NPC_SOLVER/Dataset/`.
5. Run [`solve_NP_problems_with_agent.ipynb`](./NPC_SOLVER/Codes/solve_NP_problems_with_agent.ipynb) or [`solve_NP_problems_with_agents_2.ipynb`](./NPC_SOLVER/Codes/solve_NP_problems_with_agents_2.ipynb) to reproduce the ReAct agent experiments; final saved results are in `NPC_SOLVER/AgentResults/`.
6. All outputs (evaluation tables, feature-importance plots, and training curves) are already preserved in the notebooks and can be reviewed directly on GitHub without re-execution.

## Notes on Scope

- This was a three-person team project; the report's contributions section credits Babak Hosseini Mohtasham with analyzing the base LLM's performance and training and evaluating the fine-tuned (SFT/GRPO) models, while dataset/verifier creation and agent development were shared or led by teammates — see the report for the full breakdown.
- The agent notebooks (`solve_NP_problems_with_agent.ipynb` and `..._with_agents_2.ipynb`) reflect substantial iterative prompt and tool-design experimentation across several agent versions; the final reported agent results correspond to the last iteration described in the report, not every intermediate version present in the notebooks.
- Hamiltonian Path was selected for fine-tuning and agent development specifically because baseline analysis identified it (along with Hamiltonian Cycle) as showing the steepest accuracy drop-off as graph size increased, making it the most informative test case for improvement techniques.
- The `BITCOIN_PRICE_PREDICTOR` folder in this repository is a separate, unrelated project from the same course and is not covered by this README.
