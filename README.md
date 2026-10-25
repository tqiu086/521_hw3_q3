# 521_hw3_q3

(a)
(i) 
| model           | clean | l_inf | l_2   | l_1   | union |
|-----------------|-------|------|------|------|-------|
| base            | 78.7  | 48.5 | 60.3 | 23.2 | 23.1  |
| linf fine-tuned | 77.9  | 44.6 | 57.6 | 19.5 | 19.4  |
| l2 fine-tuned   | 83.0  | 31.9 | 60.2 | 28.3 | 24.5  |
| l3 fine-tuned   | 84.7  | 0.0  | 0.1  | 26.2 | 0.0   |

$l1$ is the weakest for the base, $l\infty$ and $l2$ models (19.5–28.3%). For the $l2$ model it is $l\inf$, at 0.0% ($l2$ is nearly as bad, at $0.1%$).

(ii)
From the results, I observed fine-tuning the base on $l\infty$ lowered its $l\infty$ accuracy slightly, from $48.5$ to $44.6$.
- $l\infty$ and $l2$: these carry over to each other fairly well. The $l\infty$ model gets $57.6%$ under $l2$, and the $l2$ model gets $31.9%$ under $l\infty$.
- $l1$: neither of those gives $l1$ robustness (about $20–28%$, against $45–60%$ on their own norms).
- $l1$ model: it does not transfer at all, so sparse $l1$ training can leave a model fully broken under $l\infty$ and $l2$.
- Union: no single-norm model gets above $24.5%$.
Then, I conclude robustness transfers only partly, and unevenly.

(b)
(i)
| Model            | Clean | l_inf | l_2  | l_1  | Union | Train time / epoch |
|------------------|-------|-------|------|------|-------|--------------------|
| Base (l_inf AT)  | 78.7  | 48.5  | 60.3 | 23.2 | 23.1  | -                  |
| l_inf fine-tuned | 77.9  | 44.6  | 57.6 | 19.5 | 19.4  | ~15 s              |
| l_2 fine-tuned   | 83.0  | 31.9  | 60.2 | 28.3 | 24.5  | ~15 s              |
| l_1 fine-tuned   | 84.7  | 0.0   | 0.1  | 26.2 | 0.0   | ~17 s              |
| RAMP             | 78.5  | 41.7  | 62.7 | 44.9 | 39.6  | ~36 s              |

- RAMP has the highest union accuracy: $39.6%$, against $24.5%$ for the best single-norm model ($l2$).
- The $l1$ model has the highest clean accuracy ($84.7%$), then $l2$ ($83.0%$). RAMP’s clean accuracy is about the same as the base.
- RAMP gets decent $l1$ robustness ($44.9%$), keeps most of the $l\infty$ robustness ($41.7%$), and $l2$ improves slightly to $62.7%$.

(ii)
RAMP takes about 2.4 times as long per epoch (36 s compare with 15 s). Each batch needs two attacks ($l\infty$ and $l1$) instead of one. This is because it needs adversarial forward and backward on the adversarial sets, also an extra backward for the clean gradient and the projection for each tensor.

(c)
(i)
In my RAMP runs the union accuracy (39.6) was almost equal to the ℓ∞ accuracy (41.8). $l\infty$ is the norm that limits the union, while $l1$ ($45.0$) and $l2$ ($62.6$) are easier. RAMP uses a hard per-sample max of the two losses. I wanted a smoother way to put more weight on the weaker norm.

I then treat the two norms as groups, as in Group DRO (Sagawa et al., ICLR 2020), and keep running weights $w$ = ($w\infty$, $w_1$) over training. For each batch:
```
x_q = attack(x, linf);  x_r = attack(x, l1)      # model in eval mode
CE_q, CE_r = batch-mean cross-entropy on x_q, x_r
L = winf·CE_q + w1·CE_r + lamb·KL(p_q ‖ p_r)         # KL only on images correct under linf
w_g <- w_g * exp(eta * CE_g), normalise, clamp to [0.2, 0.8]   # eta = 0.01
update with RAMP's gradient: g = beta·GP(g_n, g_a) + (1−beta)·g_a   # beta = 0.5
```
**Updates**: RAMP’s per-sample max decides each batch independently. My weights are a memory over training that follows how weak each norm currently is. They are smoother, and they don’t depend on a single batch.

Work cited: Tramer & Boneh 2019 (MAX/AVG), Maini et al. 2020 (MSD), Li et al. 2021 (tilted ERM, a soft max) and RAMP.

(ii)
I wrote `my_loss` and a small `DROState` class (it holds the weights and the update). I trained with `adv_train(load_base_model(), my_loss, gp_beta=0.5, dro=DROState())` using the default settings, so I did not need my own copy of adv_train.

(iii)
| Model | Seed | Clean | l_inf | l_2 | l_1 | Union | s / epoch |
|---|---|---|---|---|---|---|---|
| Base | - | 78.7 | 48.5 | 60.3 | 23.2 | 23.1 | - |
| RAMP | 0 | 78.5 | 41.7 | 62.7 | 44.9 | 39.6 | 36.4 |
| RAMP | 1 | 78.5 | 41.9 | 62.5 | 45.0 | 39.6 | 36.0 |
| RAMP mean| | 78.5 | 41.8 | 62.6 | 45.0 | **39.6** | 36.2 |
| DRO+GP (mine) | 0 | 79.5 | 41.5 | 62.5 | 45.1 | 39.0 | 34.2 |
| DRO+GP (mine) | 1 | 79.1 | 40.7 | 62.6 | 44.4 | 37.9 | 35.9 |
| Mine mean| | 79.3 | 41.1 | 62.6 | 44.8 | **38.5** | 35.1 |

Same cost as RAMP (35 vs 36 s per epoch), so no equal-time RAMP run is needed. Union is 1.1 points lower on average, with 0.8 higher clean accuracy. With n = 1000 the standard error is about 1.5 points, so this is at most a weak difference.

(iv)
| Variant | Clean | l_inf | l_2 | l_1 | Union |
|---|---|---|---|---|---|
| DRO + GP (mine) | 79.5 | 41.5 | 62.5 | 45.1 | 39.0 |
| Fixed weights 0.5/0.5 (eta = 0) + GP | 79.3 | 40.7 | 62.5 | 45.9 | 38.7 |
| DRO weights, no gradient projection | 76.7 | 42.8 | 63.0 | 46.1 | 40.5 |

I changed one part of the method and ran seed 0 only. Fixed weights give 38.7, so the adaptive weights add only about 0.3 points, which is noise. Removing the projection raises union by 1.5 points and $l\infty$ by 1.3, but clean accuracy falls by 2.8 points. So the projection keeps clean accuracy up at the cost of some robustness.

(v)
The method does not improve on RAMP. 
- The two training losses are almost equal ($l\infty$ 0.70 vs $l1$ $0.62$ at epoch $10$), so the weights have little to move; $l\inf$ is the max for about 65% of samples, which a weighted sum already approximates. The weights therefore behave close to RAMP's max (and the fixed-weight ablation is within 0.3).
- The $l\infty$ and $l1$ gradients are strongly aligned (cosine 0.77 to 0.80) with equal norms, so there is no norm conflict to resolve by reweighting.
- The real problem is robust overfitting: adversarial training accuracy rises from about $50%$ to $74-78%$, while test $l\inf$ accuracy falls from $46$ to $38$ and $l1 $from $46$ to $41$ (500-image quick eval). Loss weighting does not address that. The weights themselves were not logged.
- Gradient projection never zeroes anything: $cos$($g_n$, $g_a$) is $0.52$ to $0.64$, so it only shrinks $g_n$. It acts like a clean-accuracy regulariser, which is what the ablation shows.
- $l\infty$ limits the union: union 38.5 is just below $l\infty$ 41.1.
If I have enough time, i would choose to work on weight averaging (EMA) or a self-distillation teacher against overfitting next. 

(vi)
I evaluated seed-0 models with `test_model(model, steps=200, restarts=5)`.
| Model | Clean | l_inf | l_2 | l_1 | Union |
|---|---|---|---|---|---|
| RAMP (50 x 2) | 78.5 | 41.7 | 62.7 | 44.9 | 39.6 |
| RAMP (200 x 5) | 78.5 | 41.4 | 62.7 | 43.5 | 39.0 |
| Mine (50 x 2) | 79.5 | 41.5 | 62.5 | 45.1 | 39.0 |
| Mine (200 x 5) | 79.5 | 41.2 | 62.5 | 42.7 | 37.7 |

Union falls by $0.6–1.3$ points under the stronger attack. Most of that is $l1$, which drops by $1.4–2.4$ points, while $l\inf$ and $l2$ barely move. A small drop without a collapse suggests no major gradient masking. My method only changes training and adds no randomness or input transformation, so the test-time model is unchanged and no adaptive attack is needed.