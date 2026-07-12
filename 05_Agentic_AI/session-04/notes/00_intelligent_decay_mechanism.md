# Intelligent Decay Mechanism — The "Forgetting" Layer of Agentic Memory

### Session Note: CodeSentinel — Episodic Memory Lifecycle

---

## 1. Why Do Agents Need to "Forget"?

An agent's episodic memory store grows forever if nothing prunes it. Unbounded growth causes three real problems:

- **Retrieval noise** — irrelevant old entries pollute search results
- **Latency & cost growth** — larger vector index, slower & pricier lookups
- **Staleness** — outdated facts can contaminate current answers

> **Forgetting is a feature, not a bug.** It's the mechanism that keeps memory *useful*, not just small.

This idea is directly inspired by the **Ebbinghaus Forgetting Curve** from human cognitive psychology — human memory also decays exponentially without reinforcement. We're just formalizing the same intuition for machines.

**Where this shows up in real systems:**
- **Generative Agents** (Park et al.) — popularized Recency + Relevance + Importance as a memory retrieval score
- **MemGPT / Letta** — memory hierarchy: what stays "hot" in context vs. gets paged out
- Production RAG/agent systems generally — decay keeps the vector index lean and trustworthy

---

## 2. The Core Idea

Instead of a naive **"add-all"** or **"truncate-all"** policy, we compute a **Utility Score** for every memory entry and decide: **keep, consolidate, or delete.**

Each `MemoryEntry` $M_i$ gets a composite Utility Score $S(M_i)$ built from **three factors**:

| Symbol | Name | What it measures |
|---|---|---|
| $R_i$ | **Recency** | How fresh the memory is (time-decay) |
| $E_i$ | **Relevance** | How well it matches the *current* task |
| $U_i$ | **User Utility** | How valuable a human (or the system) judged it to be |

---

## 3. Mathematical Formulation

### 3.1 Recency — Exponential Time Decay

$$R_i = e^{-\lambda(t_{current} - t_i)}$$

- $t_{current}$ — current timestamp
- $t_i$ — timestamp the memory was created
- $\lambda$ — decay rate constant (bigger $\lambda$ = faster forgetting)

**Practical trick:** instead of guessing $\lambda$, define a **half-life** $T_{1/2}$ (time after which the score halves):

$$\lambda = \frac{\ln 2}{T_{1/2}}$$

### 3.2 Relevance — Cosine Similarity

$$E_i = \text{cosine\_similarity}(\mathbf{v}_i, \mathbf{v}_{task}) = \frac{\mathbf{v}_i \cdot \mathbf{v}_{task}}{\|\mathbf{v}_i\|\,\|\mathbf{v}_{task}\|}$$

- $\mathbf{v}_i$ — embedding vector of the memory
- $\mathbf{v}_{task}$ — embedding vector of the *current* task/query

### 3.3 User Utility

$$U_i \in [0, 1] \quad \text{or} \quad U_i = N \; (\text{"retain permanently"})$$

A manually or system-assigned importance score. The special sentinel value $N$ means the memory **bypasses decay entirely** — it is never scored, never deleted.

### 3.4 Composite Utility Score

$$S(M_i) = \alpha R_i + \beta E_i + \gamma U_i$$

$\alpha, \beta, \gamma$ are **tunable hyperparameters** (commonly $\alpha + \beta + \gamma = 1$).

> A **low** $S(M_i)$ → candidate for decay.
> A **high** $S(M_i)$ → should be retained.

---

## 4. Algorithm (Pseudo-code)

```
Algorithm 1: Intelligent Decay of Episodic Memory

procedure INTELLIGENTDECAY
    M ← EpisodicMemory.get_all_entries()
    for each M_i in M do
        R_i ← CalculateRecency(M_i.Timestamp)
        E_i ← CalculateRelevance(M_i.Vector, CurrentTask.Vector)
        U_i ← M_i.UserUtility
        S(M_i) ← α R_i + β E_i + γ U_i
    end for

    for each M_i in M do
        if S(M_i) < θ_decay then
            if M_i is marked for Consolidation then
                ConsolidateToSemanticMemory(M_i)
            end if
            EpisodicMemory.Delete(M_i)
        end if
    end for
end procedure
```

**Note:** if $U_i = N$ (permanent flag), the check must **short-circuit before** scoring — otherwise a pinned memory could still mathematically fall below $\theta_{decay}$ and get deleted, defeating its purpose.

---

## 5. Worked Real Example — CodeSentinel PR Review Agent

**Scenario:** CodeSentinel has reviewed PRs for weeks. It's now reviewing a **new PR on authentication middleware**. We test whether an old memory should survive.

> **Memory $M_{47}$:** "In PR #212, flagged a SQL injection risk in `user_search.py` — raw string formatting instead of parameterized queries. Suggested `%s` placeholders with psycopg2."
> - Created **20 days ago**
> - Current task: reviewing **auth middleware** (different topic)
> - User earlier rated this finding **0.6 / 1** (useful, not pinned)

### Step 1 — Recency

Team chooses a **14-day half-life**:

$$\lambda = \frac{\ln 2}{14} = 0.0495 \text{ per day}$$

$$R_{47} = e^{-0.0495 \times 20} = e^{-0.99} \approx 0.371$$

**Reading:** decayed to ~37% of original freshness — noticeably stale, not dead.

### Step 2 — Relevance

Embeddings (toy 3-D example):

$$\mathbf{v}_{47} = [0.9,\ 0.1,\ 0.05], \qquad \mathbf{v}_{task} = [0.1,\ 0.85,\ 0.3]$$

$$\mathbf{v}_{47}\cdot\mathbf{v}_{task} = (0.9)(0.1)+(0.1)(0.85)+(0.05)(0.3) = 0.19$$

$$\|\mathbf{v}_{47}\| = \sqrt{0.9^2+0.1^2+0.05^2} \approx 0.9124$$
$$\|\mathbf{v}_{task}\| = \sqrt{0.1^2+0.85^2+0.3^2} \approx 0.9069$$

$$E_{47} = \frac{0.19}{0.9124 \times 0.9069} \approx 0.230$$

**Reading:** low relevance — SQL injection findings are not related to auth middleware review. Makes intuitive sense.

### Step 3 — User Utility

$$U_{47} = 0.6$$

### Step 4 — Composite Score

Weights: $\alpha = 0.4,\ \beta = 0.4,\ \gamma = 0.2$

$$S(M_{47}) = 0.4(0.371) + 0.4(0.230) + 0.2(0.6) = 0.1484 + 0.092 + 0.12 = 0.3604$$

### Step 5 — Decision

Threshold: $\theta_{decay} = 0.45$

$$S(M_{47}) = 0.360 < 0.45 \;\Rightarrow\; \textbf{Decay candidate}$$

Because this was a **real, correct, security-relevant finding**, it's marked `marked_for_consolidation = True`. So it is **not** deleted outright — it's **promoted to semantic memory** (e.g., "This repo's `user_search.py` has a history of SQL-injection-prone patterns") as a durable, de-timestamped fact. Only the *episodic* copy is removed.

---

## 6. How to Interpret the Result — Teaching the Diagnosis, Not Just the Number

| Score | Value | What it tells you |
|---|---|---|
| $R_i = 0.371$ | Moderate decay | Not fresh, not ancient |
| $E_i = 0.230$ | **Low — the dominant factor** | This memory is simply off-topic for *today's* task |
| $U_i = 0.6$ | Above average | The memory has real intrinsic value |
| $S = 0.360$ | Below threshold | Not useful *right now*, but not worthless |

### The Key Conclusion Pattern

> **A low $S(M_i)$ does not always mean "bad memory."** It can mean **"wrong context, right moment."**

Always ask: **which term is dragging the score down?**

| If low score is driven by... | Likely correct action |
|---|---|
| Low **relevance**, but high **user utility** | → **Consolidate to semantic memory** (fact survives, event fades) |
| Low **recency** *and* low **user utility** together | → **Just delete** (was likely noise to begin with) |
| $U_i = N$ | → **Never decay**, skip scoring entirely |

This diagnostic step — reading *why* a score is low, not just *whether* it crossed the threshold — is what separates a naive LRU-style cache eviction from truly **intelligent** forgetting.

---

## 7. Python Reference Implementation

```python
import numpy as np
from dataclasses import dataclass
from typing import Union

PERMANENT = "N"  # sentinel: retain permanently


@dataclass
class MemoryEntry:
    id: str
    timestamp: float                  # t_i
    vector: np.ndarray                # embedding
    user_utility: Union[float, str]   # U_i, float or PERMANENT
    marked_for_consolidation: bool = False


def recency_score(t_current: float, t_i: float, lam: float) -> float:
    return float(np.exp(-lam * (t_current - t_i)))


def lambda_from_half_life(half_life: float) -> float:
    return np.log(2) / half_life


def relevance_score(v_i: np.ndarray, v_task: np.ndarray) -> float:
    v_i, v_task = np.asarray(v_i, float), np.asarray(v_task, float)
    denom = np.linalg.norm(v_i) * np.linalg.norm(v_task)
    return float(np.dot(v_i, v_task) / denom) if denom else 0.0


def calculate_utility(memory, t_current, lam, v_task, alpha, beta, gamma):
    if memory.user_utility == PERMANENT:
        return {"utility_score": float("inf"), "is_permanent": True}

    R_i = recency_score(t_current, memory.timestamp, lam)
    E_i = relevance_score(memory.vector, v_task)
    U_i = float(memory.user_utility)
    S_i = alpha * R_i + beta * E_i + gamma * U_i

    return {
        "recency": round(R_i, 4),
        "relevance": round(E_i, 4),
        "user_utility": U_i,
        "utility_score": round(S_i, 4),
        "is_permanent": False,
    }


def intelligent_decay(memories, t_current, lam, v_task,
                       alpha=0.4, beta=0.4, gamma=0.2,
                       theta_decay=0.5,
                       consolidate_fn=None, delete_fn=None):
    kept, consolidated, deleted = [], [], []

    for m in memories:
        scores = calculate_utility(m, t_current, lam, v_task, alpha, beta, gamma)

        if scores["is_permanent"] or scores["utility_score"] >= theta_decay:
            kept.append({"id": m.id, **scores})
            continue

        if m.marked_for_consolidation:
            if consolidate_fn:
                consolidate_fn(m)
            consolidated.append({"id": m.id, **scores})

        if delete_fn:
            delete_fn(m.id)
        deleted.append({"id": m.id, **scores})

    return {"kept": kept, "consolidated": consolidated, "deleted": deleted}


# --- Reproduce the worked example above ---
if __name__ == "__main__":
    lam = lambda_from_half_life(14)          # 14-day half-life
    v_task = np.array([0.1, 0.85, 0.3])      # auth middleware task embedding

    m47 = MemoryEntry(
        id="M47",
        timestamp=-20,                        # 20 days before t_current
        vector=np.array([0.9, 0.1, 0.05]),
        user_utility=0.6,
        marked_for_consolidation=True,
    )

    report = intelligent_decay(
        memories=[m47],
        t_current=0,
        lam=lam,
        v_task=v_task,
        alpha=0.4, beta=0.4, gamma=0.2,
        theta_decay=0.45,
        consolidate_fn=lambda m: print(f"Consolidating {m.id} → semantic memory"),
        delete_fn=lambda mid: print(f"Deleting {mid} from episodic store"),
    )
    print(report)
    # R_47 ≈ 0.371, E_47 ≈ 0.230, S_47 ≈ 0.360 → below 0.45 → consolidated + deleted
```

---
