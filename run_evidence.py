
from MCEvidence_fixed import compare_models, compute_evidence


# =========================================================================
# CONFIGURATION - EDIT THIS
# =========================================================================

# --- Paths to the MCMC chains (root names, without the '_1.txt' suffix) ---
LCDM_ROOT = "/home/lbaldazzi/Documents/Dottorato/MCMCs/LCDM/No_H0_prior/Planck+Planck-ACT_Lensing+DESI_DR2_BAO+DESY5-Dovekie_SNeIa/PlikLiteHM-TTTEEE+Planck-ACT-Lensing+Planck-lowEE+Planck-lowTT+DESI_BAO_DR2+DESY5-Dovekie"
TPM_ROOT  = "/home/lbaldazzi/Documents/Dottorato/MCMCs/TPM/No_H0_prior/Planck+Planck-ACT-Lensing+DESI_BAO_DR2+DESY5-Dovekie/PlikHM-TTTEEE+Planck-ACT-Lensing+Planck-lowEE+Planck-lowTT+DESI_BAO_DR2+DESY5-Dovekie"

# --- Sampled physical parameters to use for the kNN manifold ---
#     (exactly the ones whose prior volume enters the evidence)
LCDM_PARAMS = ["ombh2", "omch2", "tau", "H0", "ns", "logA"]
TPM_PARAMS  = ["ombh2", "omch2", "tau", "H0", "ns", "logA",
               "c", "M", "Log_aT", "sig"]

# --- Optional: path to *.minimum or Like_stats.txt files from BOBYQA ---
#     Leave as None to fall back to the chain's sample maximum.
LCDM_LIKESTATS = '/home/lbaldazzi/Documents/Dottorato/MCMCs/LCDM/No_H0_prior/Planck+Planck-ACT_Lensing+DESI_DR2_BAO+DESY5-Dovekie_SNeIa/Like_statistics.txt'    # e.g. "/path/to/LCDM.minimum"
TPM_LIKESTATS  = '/home/lbaldazzi/Documents/Dottorato/MCMCs/TPM/No_H0_prior/Planck+Planck-ACT-Lensing+DESI_BAO_DR2+DESY5-Dovekie/Like_stats.txt'    # e.g. "/path/to/TPM.minimum"

# --- Optional: manual prior-volume overrides.  If None, the script tries
#     to read '<ROOT>.ranges' and multiply (pmax-pmin) over the params. ---
LCDM_PRIOR_VOL = None
TPM_PRIOR_VOL  = None

# --- Algorithm settings ---
KMAX    = 4         # use k = 1 .. kmax-1 nearest neighbours
BURNLEN = 0.3       # burn-in: 0..1 = fraction, >=1 = sample count
THINLEN = 0.0       # thinning factor (0 = no thinning)

# --- Scale & conventions ---
CONVENTION = "KR"        # "KR" (Kass & Raftery) or "Trotta"
BENEVENTO  = True        # also print Bayes factor with Benevento et al., 2022's sign
OUTFILE    = "/home/lbaldazzi/Documents/Dottorato/Scripts/Script_generici/Model_comparison/Results/No_H0_prior/PlikLiteHM-TTTEEE+low-TT+low-EE+Planck+ACT-Lensing+DESI_BAO_DR2+DESY5-Dovekie/evidence_run.txt"   # None = only print to stdout

# =========================================================================
# RUN
# =========================================================================

result = compare_models(
    model1={
        "root_name":    LCDM_ROOT,
        "params":       LCDM_PARAMS,
        "likestats":    LCDM_LIKESTATS,
        "prior_volume": LCDM_PRIOR_VOL,
    },
    model2={
        "root_name":    TPM_ROOT,
        "params":       TPM_PARAMS,
        "likestats":    TPM_LIKESTATS,
        "prior_volume": TPM_PRIOR_VOL,
    },
    labels               = ("LCDM", "TPM"),
    kmax                 = KMAX,
    burnlen              = BURNLEN,
    thinlen              = THINLEN,
    convention           = CONVENTION,
    benevento_convention = BENEVENTO,
    outfile              = OUTFILE,
)

# =========================================================================
# POST-PROCESS THE RETURNED VALUES
# =========================================================================
import numpy as np

print("\n" + "="*60)
print(" PROGRAMMATIC ACCESS TO RESULTS")
print("="*60)
print(f"ln(Z) for LCDM, per k:   {result['res1']['lnZ']}")
print(f"ln(Z) for TPM,  per k:   {result['res2']['lnZ']}")
print(f"ln(B_12)        per k:   {result['lnB_12']}")
print(f"log10(B_12)     per k:   {result['log10_B_12']}")
print(f"Benevento sign  per k:   {result['delta_log10_B_Benev']}")

# Example: "k-averaged" Bayes factor (usually k>=2 is more stable)
lnB_avg = np.mean(result['lnB_12'][1:])   # skip k=1 (self-NN noisiest)
print(f"\nln(B_12) averaged over k=2..{KMAX-1}: {lnB_avg:.4f}")

# Example: check against a threshold
if abs(lnB_avg) > 5.0:
    print("=> |ln B| > 5: very strong evidence.")
elif abs(lnB_avg) > 3.0:
    print("=> |ln B| > 3: strong evidence.")
elif abs(lnB_avg) > 1.0:
    print("=> |ln B| > 1: positive evidence.")
else:
    print("=> |ln B| < 1: inconclusive.")
