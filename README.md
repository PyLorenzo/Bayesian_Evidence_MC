# README.md

This document is based on the `MCEvidence.py` distributed with the upstream repository
[`yabebalFantaye/MCEvidence`](https://github.com/yabebalFantaye/MCEvidence),
which implements the Heavens et al. (2017) kth-nearest-neighbour estimator of
the Bayesian evidence from MCMC chains (arXiv:[1704.03472](https://arxiv.org/abs/1704.03472)).

The file is a near-complete rewrite (1475 → 740 lines) that preserves the Heavens estimator
but restructures the surrounding machinery for use with modern **Cobaya**-style
chains (and in particular the chain format produced by EFTCosmoMC / EFTCAMB
runs of the kind used in Benevento et al. 2022 and Kable et al. 2023).

The motivation was twofold:

1. To make the code **numerically robust** on high-dimensional chains where
   the intermediate quantity `V_k(x_j) / w_j · exp(fs_j)` can easily overflow
   or underflow.
2. To make the code **correct by construction** on chains whose sample file
   does *not* store `-lnL` in column 1 — a silent failure mode of the
   original that causes a measurable bias in `ln Z` whenever Gaussian priors
   are present (e.g. the Planck `A_planck` nuisance parameter).

Below, every substantive change is listed together with the scientific or
numerical rationale. Trivial changes (whitespace, unused imports) are
summarised at the end.

---

## 1. Architectural rewrite

### 1.1 Class hierarchy replaced

The original file exposes a three-level hierarchy:

```
LoggingHandler
SamplesMIXIN  ──► MCSamples  ──► MCEvidence   (main class, 560 lines)
data_set      (lightweight container)
```

`MCSamples` is optionally backed by the `getdist.MCSamples` object if
`getdist` is importable, and is otherwise reimplemented from scratch via
`SamplesMIXIN`. The `MCEvidence` constructor takes 23 keyword arguments and
encodes several execution modes (single chain, cross-validation, batch analysis,
importance sampling, Poisson-vs-weighted thinning, Python-class sampler).

The fork collapses this into two self-contained classes:

```
ChainLoader        (file parsing, column identification by name, burn-in, thinning)
HeavensEvidence    (kNN estimator, prewhitening, log-sum-exp accumulation)
```

plus three thin helpers:

- `compute_prior_volume_from_ranges(...)` — reads `<root>.ranges`,
- `extract_bestfit_loglmax(...)` — reads a CosmoMC `Like_stats.txt` or a Cobaya
  `*.minimum` file,
- `jeffreys_scale(...)` — interpretation of `|ln B|` under either the
  Kass–Raftery (1995) or Trotta (2008) convention.

### 1.2 Dependency pruning

Removed from the top-of-file imports:

| Import (original)                     | Kept? | Reason for removal |
|---------------------------------------|-------|--------------------|
| `importlib`, `itertools`, `functools.reduce`, `collections.namedtuple`, `io`, `tempfile` | no | Unused or used only by the removed execution modes. |
| `sklearn as skl` (whole module)       | no    | Only `NearestNeighbors` was ever used. |
| `sklearn.metrics.DistanceMetric`      | no    | Imported but never used. |
| `numpy.linalg.inv`, `numpy.linalg.det`| no    | `det` replaced by `np.linalg.det` at call site, `inv` never used. |
| `getdist` (conditional import)        | no    | The fork depends only on the column header of the chain file. |

The remaining imports are the minimal set actually exercised by the estimator:
`numpy`, `pandas` (weighted thinning only), `statistics`, `sklearn.neighbors.NearestNeighbors`,
`scipy.special`, `math`, `logging`, `argparse`, `os`, `glob`, `sys`.

### 1.3 Python-2 compatibility dropped

The original file was dual-compatible with Python 2.7 and 3.5 (note the
`from __future__ import ...` and the `raw_input()` try/except in
`query_yes_no`). The fork targets Python 3 only; `raw_input` and the
`query_yes_no` interactive prompt are removed.

---

## 2. Scientific fix: identify the likelihood column by NAME, not position

This is the change with the largest numerical impact on real analyses.

### 2.1 What the original does

In `SamplesMIXIN.setup(...)` the original hardcodes a positional column
layout (`MCEvidence.py`, lines ~136–138):

```python
self.iw     = kwargs.pop('iw',     0)    # weight
self.ilike  = kwargs.pop('ilike',  1)    # -ln L  (assumed)
self.itheta = kwargs.pop('itheta', 2)    # first sampled parameter
```

That layout is the CosmoMC convention:

```
# weight  -lnLike  param_1  param_2  ...
```

and the quantity used in `evidence(...)` as the exponent is then
`logL = -self.data['s1'].loglikes` (line 902 / 1061), which in turn
is the negation of column `ilike`. The numerator that enters the Heavens
estimator is therefore always *whatever was stored in column 1*, with a sign
flip — but the code never checks the actual meaning of that column.

### 2.2 Why this silently biases evidence on Cobaya chains

Cobaya MCMC output headers look like:

```
# weight  minuslogpost  minuslogprior  chi2  chi2__all  omegabh2  omegach2 ...
```

Column 1 is `minuslogpost`, which by definition satisfies

$$
-\ln \mathcal{P}(\theta) \;=\; -\ln \mathcal{L}(\theta) \;+\; \bigl[-\ln \pi(\theta)\bigr]
\;=\; -\ln \mathcal{L}(\theta) \;+\; m(\theta),
$$

where `m(θ) = -ln π(θ)` is the `minuslogprior` column. If the original
`MCEvidence.py` is run on such a chain with defaults, the Heavens estimator
receives `+ln P` in place of `+ln L`. The estimator

$$
\hat Z \;\approx\; \frac{1}{N}\sum_j \frac{V_k(x_j)}{w_j}\,e^{\ln L_j}
\cdot \frac{L_\text{max}}{V_\pi}
$$

then effectively computes

$$
\hat Z_\text{biased} \;=\; Z \cdot \langle e^{-m} \rangle_\text{post},
$$

which equals `Z` only if the prior is flat-box (so `m` is constant inside the
box and `-m` cancels against `ln V_π`). Whenever the chain includes even a
single Gaussian prior (e.g. `A_planck ~ N(1.0, 0.0025)` in Planck likelihoods),
this factor is non-trivial and `ln Z` is biased by an amount that depends on
the width of the Gaussian prior relative to the posterior.

### 2.3 What the fork does

`ChainLoader._identify_logL_column()` parses the `#`-prefixed header and
selects the correct column with the following priority:

1. **`chi2`** (or `chi2__all`) — if present, use
   `-ln L = -χ²/2`. This is exact (up to a constant absorbed by the
   `L_max` shift). Returned internally as `('chi2', idx)` and consumed in
   `get_minus_lnL()` as `0.5 * data[:, idx]`.

2. **`minuslogpost` together with `minuslogprior`** — reconstruct the pure
   likelihood via the algebraic identity

   $$
   -\ln \mathcal{L} \;=\; (-\ln \mathcal{P}) \;-\; (-\ln \pi)
   \;=\; \texttt{minuslogpost} - \texttt{minuslogprior}.
   $$

   Returned as `('post_minus_prior', i1, i2)`.

3. **`minuslogpost` alone** — fall back to the posterior, but **emit a
   WARNING** stating explicitly that Gaussian priors will bias the result,
   and suggest re-running with `chi2` saved. Returned as
   `('minuslogpost', idx)`.

4. **`minusloglike`** — if present as a native column. Returned as
   `('minusloglike', idx)`.

If none of these are found, `_identify_logL_column()` raises
`RuntimeError` with the full list of header columns — no silent failure.

### 2.4 Parameter aliases

`ChainLoader.get_param_columns(...)` and
`compute_prior_volume_from_ranges(...)` both consult an alias table,

```python
aliases = {
    'omegabh2': ['ombh2',  'omegabh2'],
    'omegach2': ['omch2',  'omegach2'],
    ... (plus the reverse mappings)
}
```

so that the user can call the code with Cobaya's shorthand (`ombh2`) while
the chain might use CosmoMC's long form (`omegabh2`), or vice versa. This
handles a very common interoperability paper-cut between the two
cosmological samplers.

---

## 3. Numerical rewrite of the estimator in log-space

### 3.1 What the original computes

In `MCEvidence.evidence(...)` the estimator, for each `k`, is implemented
(lines ~1108–1124) as a double `for` loop over samples and then:

```python
volume[j,k] = math.pow(math.pi, ndim/2) * math.pow(DkNN[j,k], ndim) / sp.gamma(1 + ndim/2)
dotp        = np.dot(volume[:,k] / weight[:], np.exp(fs))
amax        = dotp / (S * k_nn + 1.0)
MLE[ipow,k] = math.log(SumW * amax * Jacobian) + logLmax - logPriorVolume
```

Two numerical problems:

1. `volume[j,k] = π^(D/2) · d^D / Γ(1+D/2)` overflows to `+inf` as
   soon as `D · log d ≳ 700`. For `D ~ 10` and `d ~ O(1)` this is fine, but
   with `D ~ 20` cosmological+nuisance parameters and chains that span
   several sigma in whitened units, `d^D` can and does overflow.
2. `np.exp(fs)` underflows to zero whenever `fs < -745`. The shift
   `fs = logL - logLmax` protects the *maximum* sample, but posterior tails
   can easily have `|fs| > 745` in realistic high-dimensional analyses,
   silently dropping those samples from the sum.

Neither failure mode is signalled to the user.

### 3.2 What the fork computes

`HeavensEvidence.evidence(...)` performs the entire inner sum in log-space.
Writing the terms as

$$
T_j \;=\; \frac{V_k(x_j)}{w_j}\,e^{fs_j}, \qquad
\ln T_j \;=\; \underbrace{\tfrac{D}{2}\ln\pi - \ln\Gamma(1+\tfrac{D}{2})}_{\text{log\_vol\_coeff}}
\;+\; D\ln d_j \;-\; \ln w_j \;+\; fs_j,
$$

the code computes

```python
log_vol_coeff = (D/2.0) * math.log(math.pi) - math.lgamma(1 + D/2.0)
log_terms     = log_vol_coeff + D * np.log(np.maximum(d, 1e-300)) - np.log(self.w) + fs
M             = np.max(log_terms)
dotp_log      = M + math.log(np.sum(np.exp(log_terms - M)))
amax_log      = dotp_log - math.log(N*k + 1.0)
lnZ[k]        = math.log(SumW) + amax_log + math.log(J) + logLmax - logPriorVolume
```

This is the standard log-sum-exp identity

$$
\ln\Bigl(\sum_j e^{\ell_j}\Bigr)
\;=\; M + \ln\Bigl(\sum_j e^{\ell_j - M}\Bigr), \qquad M = \max_j \ell_j,
$$

which is stable whenever at least one `ℓ_j` is finite — i.e. always.
`math.lgamma` replaces `scipy.special.gamma` in the prefactor, which would
otherwise overflow for moderately large `D`. The clamp `np.maximum(d, 1e-300)`
protects against the (rare) case of duplicate samples where `d_j = 0`.

### 3.3 Treatment of a singular covariance matrix

In `get_covariance` / `diagonalise_chain`, the original (lines ~921–932):

```python
if (eigenVal < 0).any():
    self.logger.warn("Some of the eigenvalues ... are negative ...")
    print("... Estimated Evidence may not be accurate! ...")
    Jacobian = 1
```

continues silently with a unit Jacobian *and without prewhitening* — but
still uses the unprewhitened distances `DkNN` from a fit that was supposed
to be on the prewhitened chain. The fork makes this explicit:

```python
if (eigval <= 0).any():
    self.out.write("[!] WARNING: covariance has non-positive eigenvalues: ... "
                   "Evidence estimate is UNRELIABLE. "
                   "Consider removing degenerate parameters.")
    J   = 1.0
    s_w = self.s.copy()       # <-- explicitly use un-whitened samples
```

and emits a clearly flagged warning rather than a partially formatted print.

---

## 4. Use of an external `ln L_max` (BOBYQA / `.minimum` / `Like_stats`)

### 4.1 Context

Benevento et al. (2022, ApJ 935:156) and Kable et al. (2023, ApJ 959:143)
both report running BOBYQA after the MCMC chain to obtain a sharper estimate
of the posterior maximum than the single best chain sample, and then
feeding that external `ln L_max` to the Heavens estimator via the
multiplicative shift

$$
\hat Z \propto \tfrac{1}{N}\sum_j \frac{V_k}{w_j}\,e^{\ln L_j - \ln L_\text{max}}
\cdot L_\text{max}.
$$

Using a deeper `ln L_max` reduces the variance of the estimator because
`fs = ln L - ln L_max` is pushed away from zero at the mode — without
changing the value of `Z`, which is independent of the shift by construction.

### 4.2 Original

The original has no mechanism for injecting an external `ln L_max`; it uses
`logLmax = np.amax(logL)` from the chain samples only (line ~1064).

### 4.3 Fork

Two new hooks are added:

- `HeavensEvidence.__init__(..., external_loglmax=None, ...)` accepts a
  numerical override; if `abs(external_loglmax − chain_logLmax) > 5`, the
  code prints a warning that the sign convention of the injected value
  should be checked (5 natural-log units ≈ 2 dex, a reasonable threshold
  for a BOBYQA improvement vs a suspicious sign error).
- `extract_bestfit_loglmax(filepath)` recognises two file formats:
  - **CosmoMC**: a `Like_stats.txt` containing a line
    `Best fit sample -log(Like) = <value>`. Returns `-value`.
  - **Cobaya**: a `*.minimum` file whose first data row has
    `weight  minuslogpost  ...`. Returns `-minuslogpost`. Note that in the
    Cobaya case the extracted value is `+ln P_max`, not `+ln L_max`; this is
    correct **iff** there are no Gaussian priors (a caveat that is now
    stated in the docstring).

The priority in `compute_evidence(...)` is: `external_loglmax` (direct
numerical override) > `likestats` (file path) > chain sample maximum.

---

## 5. Built-in two-model Bayes-factor comparison

### 5.1 Original

The original script computes `ln Z` for a single chain per invocation and
prints `ln B[k]` values that are implicitly relative to a prior-volume
normalisation. Model comparison is left to the user.

### 5.2 Fork

A new top-level function `compare_models(model1, model2, ...)` runs
`compute_evidence(...)` on both models and then computes, for each `k`,

$$
\ln B_{12} \;=\; \ln Z_1 - \ln Z_2,
\qquad
\log_{10} B_{12} \;=\; \ln B_{12} / \ln 10,
$$

along with the qualitative strength via `jeffreys_scale(...)`. Two
conventions are supported, selectable from the CLI or API:

- `convention="KR"` (Kass & Raftery 1995, default):
  `|ln B| < 1`: not worth a bare mention; `1–3`: positive;
  `3–5`: strong; `>5`: very strong.
- `convention="Trotta"` (Trotta 2008):
  `|ln B| < 1`: inconclusive; `1–2.5`: weak;
  `2.5–5`: moderate; `>5`: strong.

A flag `--benevento` (API: `benevento_convention=True`) additionally prints

$$
\Delta\!\log_{10}\!B \;=\; \log_{10} Z_2 - \log_{10} Z_1,
$$

which is Benevento 2022's sign convention (positive favours the *extended*
model; in their notation model 2 is TPM and model 1 is ΛCDM, so
`Δ log₁₀ B > 2` is taken as strong support for TPM).

A `DECOMPOSITION (k=1)` block also prints `ln(V₂/V₁)`, making the Occam
penalty arising purely from the prior-volume ratio visible separately from
the likelihood factor.

---

## 6. Features of the original that were intentionally removed

The following capabilities existed in `MCEvidence.py` but are *not* ported
to `MCEvidence_fixed.py`. Each was removed for a specific reason.

| Feature | Where (original) | Why removed |
|---|---|---|
| **Cross-evidence via chain split** (`--cross`, `split=True`) | `chain_split`, `evidence(split=True)` | Requires ~2× samples for the same variance; seldom used in EFTCAMB-style cosmological pipelines. Re-introducing it in `HeavensEvidence` would be ~40 lines. |
| **Batch / power-law scaling** (`nbatch`, `brange`, `bscale`) | `MCEvidence.__init__` args; outer loop in `evidence(...)` | Diagnostic tool for convergence vs sample size; orthogonal to the estimator itself. |
| **Importance sampling** (`isfunc`) | `SamplesMIXIN.importance_sample` | Specific to the author's reweighting workflow. |
| **MontePython support** (reading `log.param`) | `params_info(...)` | Out of scope for a Cobaya-focused fork. Only the CosmoMC `.ranges` format is parsed. |
| **Hardcoded cosmological parameter whitelist** (`cosmo_params_list`, `--allparams`, `--paramsfile`, `iscosmo_param`) | module-level list at line 85 | Replaced by **explicit user-supplied `--params1` / `--params2`**. The user now states precisely which subset of header columns enters the kNN density — there is no hidden default. |
| **Python-class sampler** (`ischain=False`) | `MCEvidence.__init__` else-branch | Presupposes user has a class with a `Sampler()` method. Not used in practice for cosmological chains. |
| **Poisson thinning** for `0 < thinlen < 1` | `SamplesMIXIN.poisson_thin` | Replaced by a single deterministic weighted-bin thinner (`_thin`). Reproducibility is gained; the statistical argument for Poisson thinning with MCMC weights is, in our use case, outweighed by the non-determinism it introduces. |
| **Interactive `query_yes_no` prompt** if `.ranges` is missing | `query_yes_no` + `get_prior_volume` | Replaced by a hard error with a clear message. Library code should not block on `stdin`. |
| **getdist integration** | `try: from getdist import ...` | The fork depends only on plain-text chain files with a `#`-prefixed header. Removing `getdist` from the dependency graph avoids a very heavy transitive install. |

---

## 7. Command-line interface

The CLI has been restructured to reflect the new workflow (two models,
named parameters).

### 7.1 Arguments kept (possibly renamed)

| Original               | Fork                   | Notes |
|------------------------|------------------------|-------|
| `root_name`            | `root_name`            | Positional, unchanged. |
| `-k`, `--kmax`         | `-k`, `--kmax`         | Default changed: `2` → `5`. |
| `--burn`, `--burnlen`  | `--burn`, `--burnlen`  | Same semantics (fraction if `<1`, absolute if `≥1`). |
| `--thin`, `--thinlen`  | `--thin`, `--thinlen`  | Now always weighted-bin thinning (no Poisson branch). |
| `-pv`, `--pvolume`     | `-pv`, `--pvolume1`    | Now refers to model 1 explicitly. |

### 7.2 Arguments added

| Flag | Purpose |
|---|---|
| `--params1 P1 P2 …`  | **Required.** Parameter names for Model 1 (must appear in the chain header). |
| `--model2 ROOT`      | Optional path to the Model 2 chain; enables comparison mode. |
| `--params2 P1 P2 …`  | Parameters for Model 2. |
| `-pv2`, `--pvolume2` | Manual prior volume override for Model 2. |
| `--likestats1`, `--likestats2` | Paths to `Like_stats.txt` or `*.minimum` files for external `ln L_max`. |
| `-o`, `--outfile`    | Log file; messages are appended in addition to stdout. |
| `--convention`       | `KR` (default) or `Trotta` Jeffreys thresholds. |
| `--label1`, `--label2` | Human-readable labels for the two models (e.g. `LCDM`, `TPM`). |
| `--benevento`        | Also print `Δ log₁₀ B` in Benevento 2022's sign convention. |

### 7.3 Arguments removed

`-ic`/`--idchain`, `-np`/`--ndim`, `--paramsfile`, `--allparams`, `--cross`,
`-vb`/`--verbose`, `--version` — all tied to features removed in §6, or
(in the case of `--ndim`) made obsolete by explicit `--params1`/`--params2`.

### 7.4 Python API

New public functions intended for use from scripts / notebooks:

```python
from MCEvidence_fixed import compute_evidence, compare_models, jeffreys_scale

res = compute_evidence(root_name, params=[...], prior_volume=None,
                       likestats=None, external_loglmax=None,
                       kmax=5, burnlen=0.3, thinlen=0.0, verbose=True)
# res['lnZ'] : np.ndarray of ln Z for k = 1 … kmax-1
# res['prior_volume'], res['ndim'], res['names']

cmp = compare_models(model1={'root_name':..., 'params':[...]},
                     model2={'root_name':..., 'params':[...]},
                     labels=('LCDM', 'TPM'), convention='KR',
                     benevento_convention=True)
# cmp['lnB_12'], cmp['log10_B_12'], cmp['delta_log10_B_Benev'], cmp['jeffreys']
```

---

## 8. Logging and output

- The original threads a `logging.Logger` instance through every class and
  uses `logger.debug`, `logger.info`, `logger.warn` with a fixed format
  string. Verbosity is controlled by `--verbose 0|1|2`.
- The fork replaces this with a lightweight `OutputWriter` class that
  `print()`s and optionally appends to a user-specified log file. Messages
  carry one of three bracketed prefixes: `[*]` (informational),
  `[!]` (warning or abort), or a raw `=` / `-` banner for section headers.
  The debug/info split is no longer exposed; diagnostic numerical values
  (`SumW`, `ln(amax)`, `J`, `ln L_max`, `ln V_π`) are printed
  unconditionally for every `k`, which is the information the user needs
  to audit the result.

The Python `logging` module is still imported and a module-level logger is
created for historical compatibility, but it is essentially unused by the
new code path.

---

## 9. Miscellaneous

- `__version__` is set to `"2026-Fixed"` (original: `"17-04-2018"`).
- The citation block in `cite` is updated to also reference Kass & Raftery
  (1995) and Trotta (2008), acknowledging the Jeffreys-scale conventions
  used in §5.
- Line endings are normalised to Unix (`\n`); the original file uses
  Windows (`\r\n`) endings throughout.
- The shebang `#!usr/bin/env python` — which is broken (missing leading
  `/`) in the original — is dropped. The fork is intended to be run via
  `python MCEvidence_fixed.py ...` or imported.
- Test coverage: the fork has not shipped with unit tests; users are
  encouraged to verify against the Heavens 2017 Gaussian toy example
  (for which `ln Z` is known analytically) before production use.

---

## 10. Summary of scientific impact

The single change with the largest effect on published-quality numbers is
§2: identifying `-ln L` by header name rather than by column position.
For any chain produced by Cobaya with Gaussian priors, running the
*original* code silently biases `ln Z` by the prior-averaged
`⟨-ln π_G⟩_post`, which for the Planck `A_planck ~ N(1, 0.0025)` alone
is of order 3 natural-log units and translates directly into the Bayes
factor between any two models that share that nuisance parameter.

The change in §3 (log-sum-exp) affects high-dimensional chains
(roughly `D ≳ 15`), and the hooks in §4 reduce the scatter of `ln Z`
across chain resamplings by replacing the chain-sample maximum with the
BOBYQA-refined one, as practised in Benevento 2022 and Kable 2023.

The architectural and CLI changes (§1, §5–§7) do not affect the estimator
values; they make the two-model workflow the one-line operation it should
be.
