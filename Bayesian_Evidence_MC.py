

from __future__ import absolute_import, print_function
import os, glob, sys, math
import numpy as np
import pandas as pd
import statistics
from sklearn.neighbors import NearestNeighbors
import scipy.special as sp
import logging
from argparse import ArgumentParser

FORMAT = "%(levelname)s:%(filename)s.%(funcName)s():%(lineno)-8s %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

__status__ = "Development"
__version__ = "2026-Fixed"

desc = ('MCEvidence (fixed): log Bayesian Evidence via kth-NN (Heavens 2017). '
        'Parameter selection by name from chain header.')
cite = ('Heavens et al. (2017) - Marginal Likelihoods from MCMC - arXiv:1704.03472\n'
        'Kass & Raftery (1995); Trotta (2008) for Jeffreys scale.')


# =====================================================================
# CHAIN LOADER: parses the header, finds the sampled parameters by NAME
# =====================================================================
class ChainLoader:
    """Loads a Cobaya/CosmoMC-style chain and exposes columns by NAME."""

    def __init__(self, root_name, idchain=0, burnlen=0.0, thinlen=0.0,
                 out_writer=None):
        self.root_name = root_name
        self.out = out_writer or _DummyWriter()
        self.burnlen = burnlen
        self.thinlen = thinlen

        # Locate files
        if os.path.isfile(root_name):
            flist = [root_name]
        else:
            if idchain > 0:
                flist = (glob.glob(root_name + f'_{idchain}.txt')
                         + glob.glob(root_name + f'.{idchain}.txt'))
            else:
                flist = (glob.glob(root_name + '_*.txt')
                         + glob.glob(root_name + '.*.txt'))
        if not flist:
            raise FileNotFoundError(f"No chain files found for root: {root_name}")
        flist.sort()
        self.flist = flist
        self.out.write(f"[*] Found {len(flist)} chain file(s) for root '{root_name}'")

        # Parse the header of the first file to get column names
        self.header_cols = self._parse_header(flist[0])
        self.out.write(f"[*] Chain has {len(self.header_cols)} columns. "
                       f"First 5: {self.header_cols[:5]}")

        # Column index of weight and of -lnL (or minuslogpost as fallback)
        self.iw = self._find_col(['weight'])
        self.ilnl = self._identify_logL_column()

        # Load and concatenate
        chains = [np.loadtxt(f) for f in flist]
        # apply burn-in per chain
        if burnlen > 0:
            chains = [self._burn(c, burnlen) for c in chains]
        self.data = np.concatenate(chains, axis=0)
        # thinning (weighted)
        if thinlen > 0:
            self.data = self._thin(self.data, thinlen)

        self.nsamples = self.data.shape[0]
        self.out.write(f"[*] Total samples after burn-in/thinning: {self.nsamples}")

    def _parse_header(self, filepath):
        with open(filepath) as f:
            first = f.readline().strip()
        if not first.startswith('#'):
            raise RuntimeError(f"Chain file {filepath} has no '#'-prefixed header. "
                               "Cannot identify columns by name.")
        return first.lstrip('#').split()

    def _find_col(self, candidates):
        for c in candidates:
            if c in self.header_cols:
                return self.header_cols.index(c)
        return None

    def _identify_logL_column(self):
        """Return the column index to use as '-lnL' for the Heavens estimator.

        Preference order:
           1. If 'chi2' is present -> use -0.5*chi2 (true -lnL up to const).
              We signal this by returning ('chi2', idx).
           2. Else if 'minuslogpost' and 'minuslogprior' are present ->
              -lnL = -minuslogpost - (-minuslogprior) = ln(pi) - ln(post),
              i.e. use -(minuslogpost - minuslogprior).
           3. Else use 'minuslogpost' directly (WARNING: includes prior).
           4. Else use 'minusloglike' or 'chi2_true'.
        """
        if 'chi2' in self.header_cols:
            idx = self.header_cols.index('chi2')
            self.out.write(f"[*] Using 'chi2' column (idx={idx}): -lnL = -chi2/2")
            return ('chi2', idx)
        if 'chi2__all' in self.header_cols:
            idx = self.header_cols.index('chi2__all')
            self.out.write(f"[*] Using 'chi2__all' column (idx={idx}): -lnL = -chi2__all/2")
            return ('chi2', idx)
        if ('minuslogpost' in self.header_cols
                and 'minuslogprior' in self.header_cols):
            i1 = self.header_cols.index('minuslogpost')
            i2 = self.header_cols.index('minuslogprior')
            self.out.write(f"[*] Reconstructing -lnL from "
                           f"minuslogpost (idx={i1}) - minuslogprior (idx={i2})")
            return ('post_minus_prior', i1, i2)
        if 'minuslogpost' in self.header_cols:
            idx = self.header_cols.index('minuslogpost')
            self.out.write(f"[!] WARNING: only 'minuslogpost' available (idx={idx}). "
                           "This INCLUDES Gaussian priors and will bias the evidence "
                           "if the chain has Gaussian priors (e.g. A_planck). "
                           "Please re-run Cobaya with chi2 saved.")
            return ('minuslogpost', idx)
        if 'minusloglike' in self.header_cols:
            idx = self.header_cols.index('minusloglike')
            self.out.write(f"[*] Using 'minusloglike' (idx={idx}) as -lnL")
            return ('minusloglike', idx)
        raise RuntimeError(f"No usable likelihood column found in chain header. "
                           f"Header was: {self.header_cols}")

    def get_weights(self):
        return self.data[:, self.iw]

    def get_minus_lnL(self):
        """Return the array '-lnL' (pure likelihood, prior-free, up to const)."""
        spec = self.ilnl
        if spec[0] == 'chi2':
            return 0.5 * self.data[:, spec[1]]
        if spec[0] == 'post_minus_prior':
            _, i1, i2 = spec
            # -lnL = -(post - prior) = -lnpost + lnprior_signed
            # where minuslogprior = -lnprior_signed (if signed prior)
            # Identity: minuslogpost = minuslogL + minuslogprior
            # => minuslogL = minuslogpost - minuslogprior
            return self.data[:, i1] - self.data[:, i2]
        if spec[0] in ('minuslogpost', 'minusloglike'):
            return self.data[:, spec[1]]
        raise RuntimeError("bug: unknown ilnl spec")

    def get_param_columns(self, param_names):
        """Return (samples[n_samples, len(param_names)], missing_list).
        Looks up each requested name in the chain header.  Also tries
        common aliases (omegabh2 <-> ombh2, etc)."""
        aliases = {
            'omegabh2': ['ombh2', 'omegabh2'],
            'ombh2':    ['ombh2', 'omegabh2'],
            'omegach2': ['omch2', 'omegach2'],
            'omch2':    ['omch2', 'omegach2'],
        }
        cols = []
        names_used = []
        missing = []
        for pn in param_names:
            candidates = aliases.get(pn, [pn])
            found = None
            for c in candidates:
                if c in self.header_cols:
                    found = self.header_cols.index(c)
                    names_used.append(c)
                    break
            if found is None:
                missing.append(pn)
            else:
                cols.append(found)
        if missing:
            self.out.write(f"[!] Could not find these parameters in chain header: "
                           f"{missing}. Available: {self.header_cols[2:15]} ...")
        s = self.data[:, cols] if cols else np.zeros((self.nsamples, 0))
        return s, names_used, cols, missing

    def _burn(self, c, burnlen):
        if burnlen < 1:
            n = int(c.shape[0] * burnlen)
        else:
            n = int(burnlen)
        return c[n:, :]

    def _thin(self, c, thinlen):
        """Weighted-bin thinning (keeps highest-weight in each bin)."""
        weights = c[:, self.iw]
        N = len(weights)
        if thinlen < 1:
            N2 = int(N * thinlen)
        else:
            N2 = N // int(thinlen)
        bins = np.linspace(-1, N, N2+1)
        ind = np.digitize(np.arange(N), bins)
        thin_ix = pd.Series(weights).groupby(ind).idxmax().tolist()
        thin_ix = np.array(thin_ix, dtype=np.intp)
        return c[thin_ix, :]


# =====================================================================
# HEAVENS kNN EVIDENCE ESTIMATOR
# =====================================================================
class HeavensEvidence:
    """Heavens et al. (2017) estimator, faithful to the original MCEvidence."""

    def __init__(self, samples, minus_lnL, weights, prior_volume,
                 external_loglmax=None, kmax=5, out_writer=None):
        """
        samples         : (N, D) array of sampled parameters (on which kNN runs)
        minus_lnL       : (N,) array of -ln L  (one per sample; pure likelihood)
        weights         : (N,) array of MCMC weights (multiplicity counts)
        prior_volume    : scalar, product of (pmax-pmin) over the D params.
        external_loglmax: optional ln(L_max) from a BOBYQA/minimize run.
        """
        self.s = np.asarray(samples, dtype=np.float64)
        self.minus_lnL = np.asarray(minus_lnL, dtype=np.float64)
        self.w = np.asarray(weights, dtype=np.float64)
        self.prior_volume = float(prior_volume)
        self.external_loglmax = external_loglmax
        self.kmax = max(2, int(kmax))
        self.out = out_writer or _DummyWriter()
        self.N, self.D = self.s.shape
        assert self.minus_lnL.shape == (self.N,)
        assert self.w.shape == (self.N,)

    def evidence(self, nproc=-1):
        """Compute ln Z for k = 1 ... kmax-1.  Returns an array of length kmax-1."""
        N, D = self.N, self.D
        kmax = self.kmax

        # ---- 1. covariance and prewhitening ----
        cov = np.cov(self.s.T)
        eigval, eigvec = np.linalg.eig(cov)
        eigval = np.real(eigval)
        eigvec = np.real(eigvec)
        if (eigval <= 0).any():
            self.out.write(f"[!] WARNING: covariance has non-positive eigenvalues: {eigval}. "
                           "Evidence estimate is UNRELIABLE. "
                           "Consider removing degenerate parameters.")
            J = 1.0
            s_w = self.s.copy()
        else:
            J = math.sqrt(np.linalg.det(cov))
            s_w = np.dot(self.s, eigvec)
            s_w = s_w / np.sqrt(eigval)[np.newaxis, :]

        # ---- 2. logL and shift ----
        logL = -self.minus_lnL  # now logL is pure +ln L up to a const
        chain_logLmax = np.amax(logL)
        if self.external_loglmax is not None:
            logLmax = float(self.external_loglmax)
            if abs(logLmax - chain_logLmax) > 5:
                self.out.write(f"[!] WARNING: external_loglmax = {logLmax:.3f} "
                               f"differs from chain max = {chain_logLmax:.3f} "
                               f"by {logLmax - chain_logLmax:.3f}. "
                               "Check sign convention of injected value.")
        else:
            logLmax = chain_logLmax
        fs = logL - logLmax

        # ---- 3. kNN distances ----
        nbrs = NearestNeighbors(n_neighbors=kmax+1, metric='euclidean',
                                leaf_size=20, algorithm='auto', n_jobs=nproc).fit(s_w)
        DkNN, _ = nbrs.kneighbors(s_w)

        # ---- 4. volume + evidence per k ----
        log_vol_coeff = (D/2.0) * math.log(math.pi) - math.lgamma(1 + D/2.0)
        # V_sphere(d) = exp(log_vol_coeff) * d**D

        SumW = np.sum(self.w)
        logPriorVolume = math.log(self.prior_volume)

        lnZ = np.zeros(kmax)
        k0 = 1  # auto mode (the sample itself is its own 0th neighbour)
        for k in range(k0, kmax):
            d = DkNN[:, k]
            # compute V[j] / w[j] * exp(fs[j]) safely
            # use logsumexp-style: V[j] = exp(log_vol_coeff + D*ln d[j])
            # ratio V[j]/w[j]*exp(fs[j]) may underflow or overflow;
            # do it in one vectorized step
            log_terms = log_vol_coeff + D * np.log(np.maximum(d, 1e-300)) - np.log(self.w) + fs
            # sum = exp(lsm) where lsm = logsumexp(log_terms)
            M = np.max(log_terms)
            dotp_log = M + math.log(np.sum(np.exp(log_terms - M)))

            # a_max = dotp / (S*k + 1)
            amax_log = dotp_log - math.log(N * k + 1.0)

            lnZ[k] = math.log(SumW) + amax_log + math.log(J) + logLmax - logPriorVolume

            # debug print
            self.out.write(
                f"  [k={k}] SumW={SumW:.3e}, ln(amax)={amax_log:.4f}, J={J:.3e}, "
                f"ln(L_max)={logLmax:.4f}, ln(V_pi)={logPriorVolume:.4f}, "
                f"ln(Z)={lnZ[k]:.4f}")

        return lnZ[1:]  # drop k=0 (never used in auto mode)


# =====================================================================
# PRIOR VOLUME
# =====================================================================
def compute_prior_volume_from_ranges(root_name, target_params, out_writer):
    """Product of (pmax - pmin) for the *target_params* found in <root>.ranges.

    Returns (volume, names_used, missing).  Only parameters present in both
    the file AND the target list contribute; missing ones are reported.
    """
    ranges_file = root_name + '.ranges'
    if not os.path.exists(ranges_file):
        out_writer.write(f"[!] ERROR: {ranges_file} not found.")
        return None, [], list(target_params)
    volume = 1.0
    names_used = []
    with open(ranges_file) as f:
        ranges_dict = {}
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    ranges_dict[parts[0]] = (float(parts[1]), float(parts[2]))
                except ValueError:
                    pass
    missing = []
    aliases = {
        'omegabh2': ['ombh2', 'omegabh2'],
        'ombh2':    ['ombh2', 'omegabh2'],
        'omegach2': ['omch2', 'omegach2'],
        'omch2':    ['omch2', 'omegach2'],
    }
    for pn in target_params:
        candidates = aliases.get(pn, [pn])
        found = False
        for c in candidates:
            if c in ranges_dict:
                pmin, pmax = ranges_dict[c]
                if pmax > pmin:
                    volume *= (pmax - pmin)
                    names_used.append(c)
                    found = True
                    break
        if not found:
            missing.append(pn)
    out_writer.write(f"[*] Prior volume over {len(names_used)} params = {volume:.6e}. "
                     f"Params used: {names_used}")
    if missing:
        out_writer.write(f"[!] Missing from .ranges: {missing}")
    return volume, names_used, missing


# =====================================================================
# LIKE-STATS PARSING (true -ln L_max from BOBYQA)
# =====================================================================
def extract_bestfit_loglmax(filepath, out_writer=None):
    """Look for 'Best fit sample -log(Like) =' in a CosmoMC-style Like_stats.txt,
    or 'minuslogpost' in a Cobaya '*.minimum' file."""
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        for line in f:
            # CosmoMC convention
            if "Best fit sample -log(Like)" in line:
                try:
                    return -float(line.split('=')[-1].strip())
                except Exception:
                    pass
            # Cobaya convention (minimum file, first two data columns after header)
            if line.strip().startswith("#") and "minuslogpost" in line:
                continue
            if line.strip() and line.strip()[0].isdigit():
                try:
                    parts = line.split()
                    # Cobaya minimum: weight, minuslogpost, ...
                    # Here we return -(-minuslogpost) = +lnpost, which includes
                    # priors - only reliable if no Gaussian priors present.
                    return -float(parts[1])
                except Exception:
                    pass
    return None


# =====================================================================
# JEFFREYS / KASS-RAFTERY SCALE
# =====================================================================
def jeffreys_scale(ln_B, labels=("Model 1", "Model 2"), convention="KR"):
    """Map |ln B| to a qualitative strength.

    convention='KR' (default, Kass & Raftery 1995):
       0..1  -> not worth a bare mention
       1..3  -> positive
       3..5  -> strong
       >5    -> very strong

    convention='Trotta' (Trotta 2008):
       0..1   -> inconclusive
       1..2.5 -> weak
       2.5..5 -> moderate
       >5     -> strong
    """
    val = abs(ln_B)
    if convention == "KR":
        if val < 1.0:    strength = "Not worth a bare mention"
        elif val < 3.0:  strength = "Positive evidence"
        elif val < 5.0:  strength = "Strong evidence"
        else:            strength = "Very strong evidence"
    else:  # Trotta
        if val < 1.0:    strength = "Inconclusive"
        elif val < 2.5:  strength = "Weak evidence"
        elif val < 5.0:  strength = "Moderate evidence"
        else:            strength = "Strong evidence"

    if val < 1.0:
        return strength
    return f"{strength} for {labels[0] if ln_B > 0 else labels[1]}"


# =====================================================================
# OUTPUT WRITER
# =====================================================================
class OutputWriter:
    def __init__(self, filepath=None):
        self.filepath = filepath
        if self.filepath:
            with open(self.filepath, 'w') as f:
                f.write("")
    def write(self, text=""):
        print(text)
        if self.filepath:
            with open(self.filepath, 'a') as f:
                f.write(str(text) + "\n")

class _DummyWriter:
    def write(self, text=""): print(text)


# =====================================================================
# HIGH-LEVEL: one-model evidence
# =====================================================================
def evidence_for_model(root_name, params, prior_volume_override=None,
                       external_loglmax=None, kmax=5,
                       burnlen=0.0, thinlen=0.0, out=None, tag=""):
    out = out or _DummyWriter()
    out.write("\n" + "="*60)
    out.write(f" {tag}: {root_name}")
    out.write("="*60)

    # 1. prior volume
    if prior_volume_override is None:
        vol, _, missing = compute_prior_volume_from_ranges(root_name, params, out)
        if vol is None or missing:
            out.write("[!] Prior volume unreliable. Aborting this model.")
            return None
    else:
        vol = prior_volume_override
        out.write(f"[*] Manual prior volume: {vol}")

    # 2. load chain
    loader = ChainLoader(root_name, burnlen=burnlen, thinlen=thinlen, out_writer=out)
    s, names_used, cols, missing = loader.get_param_columns(params)
    if missing:
        out.write(f"[!] ABORT: missing parameters in chain: {missing}")
        return None
    out.write(f"[*] kNN uses {s.shape[1]} params from chain columns {cols}: {names_used}")
    out.write(f"[*] kNN uses {s.shape[0]} samples")
    minus_lnL = loader.get_minus_lnL()
    w = loader.get_weights()

    # 3. optional external logL_max
    if external_loglmax is not None:
        out.write(f"[*] Using external ln(L_max) = {external_loglmax}")
    else:
        out.write(f"[*] Chain max ln(L) = {-minus_lnL.min():.4f} "
                  f"(chain min(-lnL) = {minus_lnL.min():.4f})")

    # 4. evidence
    h = HeavensEvidence(s, minus_lnL, w, vol,
                        external_loglmax=external_loglmax, kmax=kmax, out_writer=out)
    lnZ = h.evidence()
    out.write(f"[*] {tag}: ln(Z) by k = {lnZ}")
    return {'lnZ': lnZ, 'prior_volume': vol, 'ndim': s.shape[1], 'names': names_used}


# =====================================================================
# PYTHON API  -  for use from scripts, notebooks, or pipelines
# =====================================================================
def compute_evidence(root_name, params,
                     prior_volume=None,
                     likestats=None, external_loglmax=None,
                     kmax=5, burnlen=0.0, thinlen=0.0,
                     verbose=True, outfile=None, tag="MODEL"):
    """
    Compute the log Bayesian evidence of ONE model with the Heavens (2017)
    kNN estimator.  Pure Python API - no terminal needed.

    Parameters
    ----------
    root_name : str
        Root filename of the MCMC chain (e.g. '/path/to/chain_root').
        The loader will find '<root>_*.txt' files, or you can pass a single
        full '*.txt' filename.
    params : list of str
        Parameter names (as they appear in the '#'-header of the chain file)
        to use for the kNN density estimate.  The prior volume is the product
        of (pmax - pmin) over these parameters.
    prior_volume : float or None
        Manual override of the prior volume.  If None, the script tries to
        read '<root_name>.ranges' and multiply over `params`.
    likestats : str or None
        Path to a '*.minimum' file (Cobaya) or 'Like_stats.txt' file
        (CosmoMC) containing the BOBYQA best-fit -log(Like).  If given,
        it replaces the chain's sample maximum in the estimator shift.
    external_loglmax : float or None
        Direct numerical override of ln(L_max).  Takes precedence over
        `likestats`.  In natural log.
    kmax : int
        Use k = 1 .. kmax-1 nearest neighbours.
    burnlen : float
        If 0 < burnlen < 1, fraction of samples removed as burn-in.
        If >= 1, number of initial samples removed.
    thinlen : float
        Weighted thinning factor; 0 disables thinning.
    verbose : bool
        If True, print progress to stdout.
    outfile : str or None
        If given, also append all messages to this file.
    tag : str
        Human-readable label for this model in the output.

    Returns
    -------
    dict with keys:
        'lnZ'          : np.ndarray, shape (kmax-1,)  -- ln(Z) for k=1..kmax-1
        'prior_volume' : float
        'ndim'         : int
        'names'        : list of str       -- parameters actually used
        'logLmax'      : float             -- ln(L_max) used in the estimator
    """
    out = OutputWriter(outfile) if (outfile or verbose) else _SilentWriter()
    if not verbose and outfile is None:
        out = _SilentWriter()
    # Resolve external_loglmax: direct override > likestats file > chain max
    if external_loglmax is None and likestats is not None:
        external_loglmax = extract_bestfit_loglmax(likestats, out)
        if external_loglmax is not None:
            out.write(f"[*] Extracted ln(L_max) = {external_loglmax:.4f} from {likestats}")
    return evidence_for_model(root_name, params,
                              prior_volume_override=prior_volume,
                              external_loglmax=external_loglmax,
                              kmax=kmax, burnlen=burnlen, thinlen=thinlen,
                              out=out, tag=tag)


def compare_models(model1, model2,
                   kmax=5, burnlen=0.0, thinlen=0.0,
                   convention="KR",
                   labels=("Model 1", "Model 2"),
                   benevento_convention=False,
                   verbose=True, outfile=None):
    """
    Full model comparison in one call: computes ln(Z) for two models and
    the Bayes factor ln(B_12) = ln(Z_1) - ln(Z_2).

    Parameters
    ----------
    model1, model2 : dict
        Configuration dicts for each model.  Required keys:
            'root_name' : str
            'params'    : list of str
        Optional keys (same semantics as compute_evidence):
            'prior_volume', 'likestats', 'external_loglmax'
    kmax, burnlen, thinlen : same as compute_evidence
    convention : 'KR' or 'Trotta'
        Threshold set for the Jeffreys scale.
    labels : tuple of two strings
        Names to use when printing ("favors Model X") messages.
    benevento_convention : bool
        If True, ALSO print Delta log10(B) in the Benevento 2022 convention
        (positive => favors model 2, the extended model).
    verbose : bool
    outfile : str or None

    Returns
    -------
    dict with keys:
        'res1', 'res2'       : full per-model dicts from compute_evidence
        'lnB_12'             : np.ndarray  -- ln(Z_1) - ln(Z_2) per k
        'log10_B_12'         : np.ndarray
        'delta_log10_B_Benev': np.ndarray  -- log10(Z_2) - log10(Z_1)
                                              (Benevento's sign convention)
        'jeffreys'           : list of str -- interpretation per k
    """
    out = OutputWriter(outfile)
    out.write(f"MCEvidence fixed version {__version__}")
    out.write(f"Jeffreys-scale convention: {convention}")

    def _run(cfg, tag):
        return compute_evidence(
            root_name=cfg['root_name'],
            params=cfg['params'],
            prior_volume=cfg.get('prior_volume'),
            likestats=cfg.get('likestats'),
            external_loglmax=cfg.get('external_loglmax'),
            kmax=kmax, burnlen=burnlen, thinlen=thinlen,
            verbose=verbose, outfile=outfile, tag=tag)

    res1 = _run(model1, f"MODEL 1 = {labels[0]}")
    if res1 is None:
        raise RuntimeError(f"Evidence computation failed for {labels[0]}")
    res2 = _run(model2, f"MODEL 2 = {labels[1]}")
    if res2 is None:
        raise RuntimeError(f"Evidence computation failed for {labels[1]}")

    # Bayes factor
    K = min(len(res1['lnZ']), len(res2['lnZ']))
    lnB = np.array([res1['lnZ'][k] - res2['lnZ'][k] for k in range(K)])
    log10B = lnB / math.log(10)
    delta_log10B_Benev = -log10B  # = log10(Z_2) - log10(Z_1)

    interps = [jeffreys_scale(b, labels=labels, convention=convention) for b in lnB]

    # Pretty print
    out.write("\n" + "="*60)
    out.write(" BAYES FACTOR (Jeffreys / Kass-Raftery scale)")
    out.write("="*60)
    out.write(f"Convention: ln B_12 = ln Z({labels[0]}) - ln Z({labels[1]})")
    out.write(f"  ln B_12 > 0  -> favors {labels[0]}")
    out.write(f"  ln B_12 < 0  -> favors {labels[1]}")
    out.write("-"*60)
    for k in range(K):
        msg = (f"  k={k+1}:  ln(B_12) = {lnB[k]:+.4f}  "
               f"( log10(B) = {log10B[k]:+.4f} )  => {interps[k]}")
        out.write(msg)

    if benevento_convention:
        out.write("\n--- Benevento 2022 sign convention ---")
        out.write(f"Delta log10(B) = log10(Z[{labels[1]}]) - log10(Z[{labels[0]}])")
        out.write(f"  positive => favors {labels[1]}")
        for k in range(K):
            out.write(f"  k={k+1}: Delta log10(B) = {delta_log10B_Benev[k]:+.4f}")

    out.write("\n--- DECOMPOSITION (k=1) ---")
    out.write(f"  ln Z({labels[0]}) = {res1['lnZ'][0]:.4f}   "
              f"(ndim={res1['ndim']}, V_pi = {res1['prior_volume']:.4e})")
    out.write(f"  ln Z({labels[1]}) = {res2['lnZ'][0]:.4f}   "
              f"(ndim={res2['ndim']}, V_pi = {res2['prior_volume']:.4e})")
    if res1['prior_volume'] > 0 and res2['prior_volume'] > 0:
        out.write(f"  ln(V2/V1) = "
                  f"{math.log(res2['prior_volume']/res1['prior_volume']):.4f} "
                  f"(Occam penalty of {labels[1]})")

    out.write("\nFinished.")

    return {
        'res1': res1,
        'res2': res2,
        'lnB_12': lnB,
        'log10_B_12': log10B,
        'delta_log10_B_Benev': delta_log10B_Benev,
        'jeffreys': interps,
    }


class _SilentWriter:
    """Used when verbose=False and no outfile."""
    def write(self, text=""):
        pass


# =====================================================================
# MAIN  (CLI wrapper around compare_models / compute_evidence)
# =====================================================================
if __name__ == "__main__":
    parser = ArgumentParser(prog=sys.argv[0], description=desc, epilog=cite)
    parser.add_argument("root_name",
                        help="Root filename for Model 1 (LCDM)")
    parser.add_argument("--params1", nargs='+', required=True,
                        help="Parameter names to use for Model 1 (must be in chain header)")
    parser.add_argument("--model2", default=None,
                        help="Root filename for Model 2 (TPM)")
    parser.add_argument("--params2", nargs='+', default=None,
                        help="Parameter names for Model 2")
    parser.add_argument("-k", "--kmax", type=int, default=5,
                        help="Maximum K for k-NN (default 5)")
    parser.add_argument("--burn", "--burnlen", dest="burnlen", type=float, default=0.0,
                        help="Burn-in fraction (0 < x < 1) or number of steps (>= 1)")
    parser.add_argument("--thin", "--thinlen", dest="thinlen", type=float, default=0.0,
                        help="Thinning factor")
    parser.add_argument("-pv", "--pvolume1", type=float, default=None,
                        help="Manual override: prior volume for Model 1")
    parser.add_argument("-pv2", "--pvolume2", type=float, default=None,
                        help="Manual override: prior volume for Model 2")
    parser.add_argument("--likestats1", default=None,
                        help="Path to Model 1 *.minimum or Like_stats file")
    parser.add_argument("--likestats2", default=None,
                        help="Path to Model 2 *.minimum or Like_stats file")
    parser.add_argument("-o", "--outfile", default=None,
                        help="Path to write a log of the run")
    parser.add_argument("--convention", choices=["KR", "Trotta"], default="KR",
                        help="Jeffreys-scale convention (default KR)")
    parser.add_argument("--label1", default="Model 1",
                        help="Short label for Model 1 (e.g. LCDM)")
    parser.add_argument("--label2", default="Model 2",
                        help="Short label for Model 2 (e.g. TPM)")
    parser.add_argument("--benevento", action="store_true",
                        help="Also print Bayes factor in Benevento 2022's sign convention")
    args = parser.parse_args()

    model1 = {
        'root_name': args.root_name,
        'params': args.params1,
        'prior_volume': args.pvolume1,
        'likestats': args.likestats1,
    }
    if args.model2 is None:
        # Single-model mode
        res = compute_evidence(**model1,
                               kmax=args.kmax, burnlen=args.burnlen,
                               thinlen=args.thinlen, outfile=args.outfile,
                               tag=f"MODEL 1 = {args.label1}")
        if res is None:
            sys.exit(1)
    else:
        if not args.params2:
            print("[!] --params2 required when --model2 is given.")
            sys.exit(1)
        model2 = {
            'root_name': args.model2,
            'params': args.params2,
            'prior_volume': args.pvolume2,
            'likestats': args.likestats2,
        }
        compare_models(model1, model2,
                       kmax=args.kmax, burnlen=args.burnlen, thinlen=args.thinlen,
                       convention=args.convention,
                       labels=(args.label1, args.label2),
                       benevento_convention=args.benevento,
                       outfile=args.outfile)