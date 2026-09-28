import re
import json
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

from src.utils.latex_fmt import br_num, br_sci

LIGHT2VER = {"AB": "v1", "AM": "v9", "BR": "v10"}
LIGHT_LABEL = {
    "AB": "AB (mista, 2700K+6500K)",
    "AM": "AM (quente, 2800K)",
    "BR": "BR (branca, 6000K)",
}
MODEL = "convnext"
PRED_DIR = "results/predictions"
KEY_RE = re.compile(r"^[A-Z]+_(\d{20})_(\d+)\.npy$")
OUT_JSON = "results/friedman_fase2_provisorio.json"
OUT_TEX = "dissertacao/tables/friedman_test.tex"


def load_fardo_means(version):
    d = np.load(f"{PRED_DIR}/{MODEL}_{version}_preds.npz", allow_pickle=True)
    files, y_true, y_probs = d["y_files"], d["y_true"], d["y_probs"]
    by_fardo = {}
    for fn, yt, p in zip(files, y_true, y_probs):
        m = KEY_RE.match(str(fn))
        if not m:
            continue
        by_fardo.setdefault(m.group(1), []).append(float(p[yt]))
    means = {f: float(np.mean(v)) for f, v in by_fardo.items()}
    counts = {f: len(v) for f, v in by_fardo.items()}
    return means, counts


def desempenho_label(pos, total):
    if pos == 0:
        return "superior"
    if pos == total - 1:
        return "inferior"
    return "intermediário"


def write_tex(mean_ranks, chi2, p_val, N):
    total = len(mean_ranks)
    rows = [
        f"{LIGHT_LABEL[lt]} & {br_num(mean_ranks[lt], 2)} & {desempenho_label(i, total)} \\\\"
        for i, lt in enumerate(mean_ranks.index)
    ]
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Fonte de Luz & Posto Médio & Desempenho \\",
        r"\midrule",
        *rows,
        r"\midrule",
        (r"\multicolumn{3}{l}{\footnotesize Friedman $\chi^2 = " + br_num(chi2, 2)
         + r"$, $p = " + br_sci(p_val, 2)
         + r"$, $N_{\text{fardos}} = " + str(N) + r"$} \\"),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(lines))


def main():
    means, counts = {}, {}
    for lt, ver in LIGHT2VER.items():
        means[lt], counts[lt] = load_fardo_means(ver)
        print(f"[{lt}={ver}] fardos no teste: {len(means[lt])}")

    fardos = sorted(set(means["AB"]) & set(means["AM"]) & set(means["BR"]))
    N = len(fardos)
    print(f"N (fardos em comum nas 3 luzes): {N}")
    assert N > 30, f"N de fardos pareados baixo demais para o teste ser informativo ({N})"

    mat = pd.DataFrame({lt: [means[lt][f] for f in fardos] for lt in ["AB", "AM", "BR"]})

    chi2, p_val = friedmanchisquare(mat["AB"], mat["AM"], mat["BR"])
    assert np.isfinite(chi2) and np.isfinite(p_val)

    # convenção explícita (correcoes_banca.md item 3): maior prob. média -> melhor -> maior posto
    ranks = mat.rank(axis=1, ascending=True)  # pior=1 ... melhor=3
    mean_ranks = ranks.mean().sort_values(ascending=False)

    raw_mean = mat.mean().sort_values(ascending=False)
    assert list(mean_ranks.index) == list(raw_mean.index), "posto médio não bate com a média bruta"

    avg_patches = {lt: float(np.mean([counts[lt][f] for f in fardos])) for lt in LIGHT2VER}

    result = {
        "N": N,
        "chi2": float(chi2),
        "p_val": float(p_val),
        "mean_ranks": {c: float(v) for c, v in mean_ranks.items()},
        "avg_patches_por_fardo": avg_patches,
        "n_fardos_por_condicao_isolada": {lt: len(means[lt]) for lt in LIGHT2VER},
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    write_tex(mean_ranks, chi2, p_val, N)
    print(f"\n[OK] {OUT_TEX} escrito.")


if __name__ == "__main__":
    main()
