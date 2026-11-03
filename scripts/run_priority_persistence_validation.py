#!/usr/bin/env python3
"""Rolling-origin validation of drift-priority persistence (analysis only, training data only).

    python scripts/run_priority_persistence_validation.py [--overwrite]

Design, predictors, expected directions and the decision rule are fixed in
``quantum_drift_scheduler.analysis.priority_persistence``. The holdout is never used: the boundary
comes from timestamps alone, and every holdout row is dropped before any computation.
Writes ``results/priority_persistence_validation/`` (refuses to overwrite without ``--overwrite``).
If any integrity check fails nothing scientific is published (an INVALID marker and the audit only).
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quantum_drift_scheduler.analysis import priority_persistence as pp
from quantum_drift_scheduler.analysis.holdout import DEFAULT_TRAIN_FRACTION
from quantum_drift_scheduler.analysis.priorities import (
    PRIORITY_EPSILON,
    learn_pair_priorities,
    training_fingerprint,
)
from quantum_drift_scheduler.config import DEFAULT_CONFIG
from quantum_drift_scheduler.data.compatibility import load_compatibility_registry
from quantum_drift_scheduler.scheduling import load_recorded_trace

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DEFAULT_OUT = RESULTS / "priority_persistence_validation"
REFERENCE_SUMMARY = RESULTS / "priority_analysis_v2" / "priority_analysis_summary.json"
REFERENCE_PRIORITIES = RESULTS / "priority_analysis_v2" / "priority_results.csv"


def file_hashes(paths):
    out = {}
    for p in paths:
        out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def tree_hashes(root: Path, exclude: Path):
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and exclude not in p.parents]
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def digest(obj) -> str:
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items() if not k.startswith("_")}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return None if not math.isfinite(x) else round(float(x), 12)
        if isinstance(x, (np.integer,)):
            return int(x)
        return x
    return hashlib.sha256(json.dumps(clean(obj), sort_keys=True).encode()).hexdigest()


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def write_csv(path: Path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if (isinstance(v, float) and not math.isfinite(v)) else v for v in r])


def flip_from(shots, t0, side):
    """Copy of ``shots`` with outcomes flipped at/after ``t0`` (side='after') or before it ('before')."""
    keep = (lambda s: s.arrival_time >= t0) if side == "after" else (lambda s: s.arrival_time < t0)
    return [dataclasses.replace(s, outcome=1 - s.outcome) if keep(s) else s for s in shots]


def run_audit(train, registry, specs, window_origin, boundary, origins, tables, reference_hashes_before):
    checks = []

    def add(name, passed, detail=""):
        checks.append((name, bool(passed), detail))

    ref = json.loads(REFERENCE_SUMMARY.read_text(encoding="utf-8"))["split"]
    add("boundary_matches_production_split", math.isclose(boundary, ref["boundary"], rel_tol=0, abs_tol=1e-9),
        f"boundary={boundary!r}; recorded production boundary={ref['boundary']!r}")
    add("only_training_shots_used", all(s.arrival_time < boundary for s in train) and len(train) == ref["train_shots"],
        f"{len(train)} training shots, max time {max(s.arrival_time for s in train)!r} < {boundary!r}; "
        f"production recorded {ref['train_shots']} training shots; holdout rows dropped before any computation")
    elig = {frozenset(p.key) for p in registry.eligible_pairs()}
    add("eligibility_from_reviewer_decision_only",
        len(elig) == 29 and all(p.reviewer_decision.value == "Comparable" for p in registry.eligible_pairs())
        and all({frozenset(p) for p in o.pairs} == elig for o in origins),
        f"{len(elig)} Comparable pairs; every origin scores exactly this set")
    add("pair_order_identical_across_origins_and_baselines", len({o.pairs for o in origins}) == 1
        and all(len(v) == len(origins[0].pairs) for o in origins for v in o.scores().values()),
        "one pair list, one outcome vector per origin, all predictors aligned to it")

    leak = []
    for spec, table, o in zip(specs, tables, origins):
        past = [s for s in train if s.arrival_time < spec.origin_time]
        ok = (max(s.arrival_time for s in past) < spec.origin_time and table.metadata.train_end < spec.origin_time
              and table.metadata.boundary == spec.origin_time
              and table.metadata.training_fingerprint == training_fingerprint(past))
        leak.append(ok)
        # invariance: flipping every outcome at/after the origin must not change the priority table
        flipped, _ = pp.priority_at_origin(flip_from(train, spec.origin_time, "after"), registry, spec, window_origin)
        leak.append(flipped.priorities == table.priorities)
    add("priority_uses_only_observations_before_origin_and_ignores_future_outcomes", all(leak),
        "per origin: max past time < origin, metadata.train_end < origin, fingerprint matches, and flipping all later outcomes leaves the table identical")

    inv = []
    for spec, o in zip(specs, origins):
        a, z, _ = pp.future_transitions(flip_from(train, spec.origin_time, "before"), registry, spec)
        inv.append(np.array_equal(a, o.abs_d, equal_nan=True) and np.array_equal(z, o.z_sq, equal_nan=True))
    add("future_statistics_ignore_past_outcomes", all(inv),
        "flipping every outcome before the origin leaves future |D| and Z^2 identical")

    fut_ok = all(spec.future_end <= boundary for spec in specs) and all(
        a.future_end <= b.origin_time for a, b in zip(specs, specs[1:]))
    add("validation_intervals_disjoint_and_inside_training", fut_ok,
        "; ".join(f"[{s.origin_time:.1f},{s.future_end:.1f})" for s in specs))

    form = all(
        math.isclose(p.priority_score, 1.0 / (p.mean_abs_z + PRIORITY_EPSILON), rel_tol=1e-12)
        for t in tables for p in t.priorities if p.sufficient_data
    )
    add("production_priority_formula_unchanged", form, "score == 1/(mean|Z| + 1e-12) for every sufficient pair at every origin")
    zero = sum(1 for t in tables for p in t.priorities if not p.sufficient_data)
    add("priority_sufficiency_reported", True, f"{zero} pair-origin(s) with insufficient past data (score 0)")

    full = learn_pair_priorities(train, registry, boundary=boundary, window_origin=window_origin,
                                 train_fraction=DEFAULT_TRAIN_FRACTION, window_seconds=pp.WINDOW_SECONDS)
    with REFERENCE_PRIORITIES.open(encoding="utf-8") as fh:
        ref_scores = {frozenset((r["circuit_a"], r["circuit_b"])): float(r["priority_score"]) for r in csv.DictReader(fh)}
    same = len(ref_scores) == 29 and all(
        math.isclose(p.priority_score, ref_scores[p.key], rel_tol=1e-9) for p in full.priorities)
    add("full_training_priorities_reproduce_existing_production_output", same,
        "learn_pair_priorities at the production boundary equals results/priority_analysis_v2/priority_results.csv (29 pairs)")

    excl = [(o.spec.index, o.pairs[i]) for o in origins for i in range(len(o.pairs)) if not o.included[i]]
    add("min_future_transitions_reported", True,
        f"minimum {pp.MIN_FUTURE_TRANSITIONS} valid future transitions; {len(excl)} excluded pair-origin(s): {excl}")
    add("horizon_and_origin_counts", all(o.abs_d.shape[1] == pp.HORIZON_WINDOWS - 1 for o in origins) and len(origins) == pp.N_ORIGINS,
        f"{len(origins)} origins, {pp.HORIZON_WINDOWS - 1} future transitions per pair")

    try:
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        add("no_tracked_file_modified", status == "", "git status (tracked files) is clean" if not status else status)
    except Exception as exc:  # git unavailable: report, do not pass silently
        add("no_tracked_file_modified", False, f"could not run git: {exc}")
    return checks


def make_plots(out: Path, res, origins):
    prim = res["outcomes"]["abs_d"]
    x = np.arange(len(origins))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, (name, col, off) in enumerate((("priority", "#1b6ca8", -0.15), ("hist_mean_abs_d", "#c0392b", 0.0),
                                          ("wait_length_gap", "#7f8c8d", 0.15))):
        r = prim[name]
        y = np.array(r["per_origin"])
        ci = np.array(r["per_origin_ci"])
        ax.errorbar(x + off, y, yerr=[y - ci[:, 0], ci[:, 1] - y], fmt="o", color=col, capsize=3, label=f"{name} (expected sign {r['sign']:+d})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [f"origin {i}\nwindow {o.spec.window_index}" for i, o in enumerate(origins)])
    ax.set_ylabel("Spearman correlation with future mean |D|")
    ax.set_title("Per-origin correlation (block-bootstrap 95% CI, 4 transitions per pair)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(out / "per_origin_correlation.png", dpi=140); plt.close(fig)

    r = prim["priority"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(r["_null_T"][np.isfinite(r["_null_T"])], bins=60, color="#b8c4d0")
    ax.axvline(r["T"], color="#1b6ca8", lw=2, label=f"observed T = {r['T']:+.3f}")
    ax.axvline(r["null_T"]["q05"], color="k", ls="--", lw=1, label=f"null 5th pct = {r['null_T']['q05']:+.3f}")
    ax.set_xlabel("mean over origins of Spearman(priority, future mean |D|)")
    ax.set_ylabel("circuit-label permutations")
    ax.set_title(f"Circuit-label permutation null (one-sided p = {r['T_perm_p']:.3f})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "permutation_null_priority.png", dpi=140); plt.close(fig)

    fig, axes = plt.subplots(1, len(origins), figsize=(3.0 * len(origins), 3.2), sharey=True)
    for ax, o, rho in zip(np.atleast_1d(axes), origins, r["per_origin"]):
        ax.scatter(o.past["priority"], o.outcome("abs_d"), s=14, color="#1b6ca8")
        ax.set_title(f"origin {o.spec.index}: rho={rho:+.2f}", fontsize=9)
        ax.set_xlabel("priority at origin")
    np.atleast_1d(axes)[0].set_ylabel("future mean |D|")
    fig.tight_layout(); fig.savefig(out / "priority_vs_future_disagreement.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = [n for n in pp.PREDICTORS if pp.PREDICTORS[n] != 0]
    T = np.array([prim[n]["T"] * prim[n]["sign"] for n in names])
    ci = np.array([[prim[n]["T_ci"][0] * prim[n]["sign"], prim[n]["T_ci"][1] * prim[n]["sign"]] for n in names])
    ci = np.sort(ci, axis=1)
    ax.errorbar(T, np.arange(len(names)), xerr=[T - ci[:, 0], ci[:, 1] - T], fmt="o", color="#1b6ca8", capsize=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(np.arange(len(names)), names)
    ax.set_xlabel("aligned T = expected sign x mean Spearman (positive = expected direction)")
    ax.set_title("Predictors on identical future observations")
    fig.tight_layout(); fig.savefig(out / "baseline_comparison.png", dpi=140); plt.close(fig)


def build_report(out, res, decision, origins, specs, audit, args_note):
    prim, sec = res["outcomes"]["abs_d"], res["outcomes"]["z_sq"]
    p = prim["priority"]
    V = decision["verdict"]
    c = decision["criteria"]
    L = []
    L.append("# Rolling-origin validation of drift-priority persistence\n")
    L.append("Analysis only. Training observations only (strictly before the existing 75% boundary); no holdout outcome, "
             "no model, no scheduler and no earlier result was touched. Everything below is read from the CSV/JSON files in this directory.\n")
    L.append(f"## Verdict: **{V.upper().replace('_', ' ')}**\n")
    verdict_text = {
        "supported": "All four predeclared criteria were met. This supports persistence of historical pairwise disagreement (claim A) only.",
        "not_established": "At least one predeclared criterion failed and the confidence interval excludes a moderate persistent effect. Persistence was **not established**. This is not proof that persistence is impossible.",
        "inconclusive": "At least one predeclared criterion failed, and the confidence interval is still compatible with a moderate persistent effect (lower limit <= -0.3). The design cannot separate a real effect from noise: **inconclusive**.",
    }[V]
    L.append(verdict_text + "\n")
    L.append("| criterion (predeclared) | result |\n|---|---|")
    L.append(f"| T = mean per-origin Spearman(priority, future mean abs D) < 0 | {'met' if c['expected_direction_T_negative'] else 'NOT met'} (T = {fmt(p['T'])}) |")
    L.append(f"| 95% block-bootstrap CI of T entirely below 0 | {'met' if c['ci_excludes_zero_in_expected_direction'] else 'NOT met'} ([{fmt(p['T_ci'][0])}, {fmt(p['T_ci'][1])}]) |")
    L.append(f"| correlation < 0 in at least {pp.MIN_CONSISTENT_ORIGINS} of {len(origins)} origins | {'met' if c['consistent_across_origins'] else 'NOT met'} ({p['origins_in_expected_direction']} of {p['origins_defined']}) |")
    L.append(f"| one-sided circuit-label permutation p <= {pp.ALPHA} | {'met' if c['beats_circuit_permutation_null'] else 'NOT met'} (p = {p['T_perm_p']:.4f}) |\n")

    L.append("## What this does and does not test\n")
    L.append("- **A. Historical pairwise-drift persistence** - tested here: does a ranking learned from the past of a pair predict its later disagreement?\n"
             "- **B. Forecasting future device drift** - not tested (the model-based forecast already failed in `results/deployment_validation/`; nothing here uses it).\n"
             "- **C. Whether close scheduling causally improves comparison accuracy** - not tested; the recording has no counterfactual schedules.\n\n"
             "Success on A would not imply B or C. Direction was fixed in advance: priority vs future mean |D| expected NEGATIVE; historical mean |Z| and historical mean |D| vs future mean |D| expected POSITIVE.\n")

    L.append("## Exact design (fixed before any outcome was computed)\n")
    L.append(f"- Windows {pp.WINDOW_SECONDS:.0f} s aligned to the trace minimum, as in production. Production `learn_pair_priorities` is called unchanged, on shots strictly before each origin.\n"
             f"- {len(specs)} expanding-window origins at windows {', '.join(str(s.window_index) for s in specs)}; each future interval is {pp.HORIZON_WINDOWS} windows ({pp.HORIZON_WINDOWS - 1} transitions per pair), "
             "disjoint and contiguous; the last ends at "
             f"{specs[-1].future_end:.1f} s, before the boundary.\n"
             f"- Rationale: the training span has 36 full windows; production needs >= 10 valid transitions (11 windows) for a positive score, so the first origin cannot be earlier than window 11, leaving 25 windows. Six origins would give fewer than {pp.MIN_FUTURE_TRANSITIONS} future transitions per pair, so five origins with a {pp.HORIZON_WINDOWS}-window horizon is the largest design that keeps both rules.\n"
             f"- Minimum {pp.MIN_FUTURE_TRANSITIONS} valid future transitions per pair-origin; failing pair-origins are listed in `pair_metrics_by_origin.csv` (column `included`).\n"
             f"- Future grid starts at the origin, so no past observation can enter a future window.\n"
             f"- Inference unit is the pair at an origin; transitions and shots are not treated as independent. Bootstrap: moving blocks of {pp.BLOCK_LENGTH} transitions with the same time indices for all pairs, then a resample of origins ({res['bootstrap']['draws']} draws, seed {res['bootstrap']['seed']}). Permutation: {res['permutation']['n']} circuit-label permutations of the 9 circuits (mean {res['permutation']['mean_pairs_scored']:.1f} of 29 pairs still map to an eligible pair; minimum {res['permutation']['min_pairs_scored']}).\n")

    L.append("## Integrity and leakage audit\n")
    L.append(f"{sum(1 for a in audit if a[1])} of {len(audit)} checks passed (`integrity_audit.csv`). Notable: flipping every outcome at/after an origin leaves that origin's priority table identical; flipping every outcome before it leaves the future statistics identical; the full-training priorities reproduce `results/priority_analysis_v2/priority_results.csv`; repeated runs are deterministic; all earlier result files hash identically before and after.\n")

    L.append("## Primary result: priority vs future mean |D| (expected sign negative)\n")
    L.append("| origin | pairs | Spearman | 95% block-bootstrap CI | permutation p (one-sided) |\n|---|---|---|---|---|")
    for i, o in enumerate(origins):
        L.append(f"| {i} (window {o.spec.window_index}) | {int(o.included.sum())} | {fmt(p['per_origin'][i])} | [{fmt(p['per_origin_ci'][i][0])}, {fmt(p['per_origin_ci'][i][1])}] | {p['per_origin_perm_p'][i]:.4f} |")
    L.append(f"| **mean over origins (T)** | {int(sum(o.included.sum() for o in origins))} pair-origins | **{fmt(p['T'])}** | [{fmt(p['T_ci'][0])}, {fmt(p['T_ci'][1])}] | {p['T_perm_p']:.4f} |")
    L.append(f"\nPooled correlation over all pair-origin rows (descriptive only, scores are not comparable across origins and rows are dependent): {fmt(p['pooled_descriptive'])}. "
             f"Circuit-label null for T: mean {fmt(p['null_T']['mean'])}, SD {p['null_T']['sd']:.3f}, 5th percentile {fmt(p['null_T']['q05'])}. "
             f"(For 29 independent pairs one origin would have SD about {1/math.sqrt(28):.2f}; pairs share circuits, so the real uncertainty is larger.)\n")

    L.append("## Baselines on identical future observations\n")
    L.append("Aligned T = expected sign x T, so positive means the expected direction. Delta = aligned T(priority) minus aligned T(baseline), paired bootstrap.\n")
    L.append("| predictor | expected sign | T (raw) | aligned T | 95% CI of T | permutation p | origins in expected direction | Delta vs priority [95% CI] |\n|---|---|---|---|---|---|---|---|")
    for n in pp.PREDICTORS:
        r = prim[n]
        if r["sign"] == 0:
            L.append(f"| {n} | n/a | undefined | undefined | n/a | n/a | n/a | n/a |")
            continue
        d = r["delta_vs_priority"]
        dtxt = "reference" if n == "priority" else f"{fmt(d['estimate'])} [{fmt(d['ci'][0])}, {fmt(d['ci'][1])}]"
        L.append(f"| {n} | {r['sign']:+d} | {fmt(r['T'])} | {fmt(r['sign'] * r['T'])} | [{fmt(r['T_ci'][0])}, {fmt(r['T_ci'][1])}] | {r['T_perm_p']:.4f} | {r['origins_in_expected_direction']}/{r['origins_defined']} | {dtxt} |")
    L.append("\nInterpretation and limits of each baseline:\n"
             "- *uniform*: a constant ranking carries no information; its correlation is undefined. It is the reference for 'no ranking'.\n"
             "- *reverse_priority* and *hist_mean_abs_z*: exact rank mirrors of priority (except for zero-score pairs), so they add no independent evidence; they exist to show the direction was not reinterpreted.\n"
             "- *hist_mean_abs_d*: the same past data without noise normalisation; if it does no better than priority, the normalisation is not what carries information.\n"
             "- *wait_length_gap*: a diagnostic only. `|l_A - l_B|` determines the model-derived comparison error, so it checks whether any pattern is just pair geometry; it is not an alternative scheduler.\n"
             "- *circuit-label permutation*: keeps the shared-circuit structure and score distribution; a permuted label can map a pair onto an ineligible pair, in which case that pair is dropped from that permutation's correlation.\n")

    L.append("## Secondary outcome: future mean Z^2\n")
    s = sec["priority"]
    L.append(f"Priority vs future mean Z^2 (expected sign negative): T = {fmt(s['T'])}, 95% CI [{fmt(s['T_ci'][0])}, {fmt(s['T_ci'][1])}], permutation p = {s['T_perm_p']:.4f}, "
             f"per-origin {', '.join(fmt(v, 2) for v in s['per_origin'])}. Not used for the verdict.\n")

    L.append("## Limitations\n")
    L.append("- **Effective sample size.** 29 pairs are built from 9 circuits; each circuit appears in several pairs, so the pairs are dependent and the effective number of independent units is far below 29. The circuit-label permutation accounts for this; the bootstrap alone does not.\n"
             f"- **Few transitions.** Four future transitions per pair-origin make each pair's outcome noisy; five origins make T and its CI coarse. Transitions are also serially correlated (adjacent transitions share a window).\n"
             "- **Few origins and nested priorities.** Priorities at later origins reuse earlier data, so origins are not independent replications.\n"
             "- **Per-origin intervals are crude.** With 4 transitions and blocks of 2 there are only 3 distinct blocks, so a per-origin bootstrap interval can touch its own point estimate. They are descriptive; only the interval for T (which also resamples origins) enters the decision.\n"
             f"- **Borderline margins.** The upper limit of T's interval ({fmt(p['T_ci'][1], 4)}) and the permutation p ({p['T_perm_p']:.4f}) sit next to their thresholds (0 and {pp.ALPHA}), and the lower limit ({fmt(p['T_ci'][0], 4)}) sits next to the -{pp.MODERATE_EFFECT} line that separates 'inconclusive' from 'not established'. Reading the label as sharper than the margins is not justified. The rule was fixed beforehand and was not adjusted.\n"
             "- **Pooled correlation** mixes scores from different origins and is descriptive only.\n"
             "- **Scope.** Only the nine low-wait-length circuits that appear in `Comparable` pairs are involved. The recording contains one device and one drift realisation.\n"
             "- **Training portion only.** The 25% holdout was not used, and it has in any case been inspected in earlier diagnostics, so it cannot serve as a pristine confirmation set.\n")

    L.append("## Recommendation\n")
    if V == "supported":
        L.append("Historical pairwise disagreement ranking persisted over time in this data. That is evidence for A only; it does not show that the model forecasts drift (B) or that closer scheduling improves accuracy (C), and it needs replication on independent data before any scheduling decision rests on it.\n")
    else:
        L.append("Do not rely on the existing drift-priority ranking to order work until new evidence is available: this validation did not establish that it carries persistent information. "
                 f"{'The result is also underpowered for a moderate effect, so a larger or independent dataset is needed to decide.' if V == 'inconclusive' else 'The interval excludes a moderate persistent effect in this data, but a small effect cannot be ruled out.'}\n")
    L.append("## Next scientifically justified step\n")
    L.append("Obtain independent data for confirmation (a new recording, or a fresh pre-registered holdout) and repeat this exact predeclared protocol with more future transitions per pair. If persistence is still not established there, drop the priority term from any scheduler design rather than tuning it.\n")
    L.append("## Files\n")
    L.append("`origin_design.csv`, `pair_metrics_by_origin.csv`, `priority_correlations.csv`, `bootstrap_intervals.csv`, `permutation_null.csv`, `baseline_comparison.csv`, `integrity_audit.csv`, `summary.json`, plots `per_origin_correlation.png`, `permutation_null_priority.png`, `priority_vs_future_disagreement.png`, `baseline_comparison.png`.\n")
    L.append("## Reproduce\n```\npython scripts/run_priority_persistence_validation.py --overwrite\npython -m pytest tests/test_priority_persistence.py -q\n```\n")
    (out / "experiment_report.md").write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    out: Path = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        print(f"{out} exists; pass --overwrite to replace it", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    before = tree_hashes(RESULTS, out)
    inputs_before = file_hashes([DEFAULT_CONFIG.default_dataset_path, DEFAULT_CONFIG.compatibility_registry_path])

    registry = load_compatibility_registry()
    trace = load_recorded_trace()
    times = [s.arrival_time for s in trace]
    t0, t1 = min(times), max(times)
    boundary = t0 + DEFAULT_TRAIN_FRACTION * (t1 - t0)  # same rule as split_trace, from timestamps only
    train = [s for s in trace if s.arrival_time < boundary]
    n_dropped = len(trace) - len(train)
    del trace, times  # holdout rows are never referenced again

    specs = pp.build_origin_schedule(t0, boundary)
    origins = pp.build_origin_data(train, registry, specs, t0)
    tables = [pp.priority_at_origin(train, registry, s, t0)[0] for s in specs]
    circuits = registry.eligible_circuits()
    print(f"{len(train)} training shots ({n_dropped} holdout rows discarded unread); {len(origins)} origins; running statistics ...")
    res = pp.analyze(origins, circuits=circuits)
    res_repeat = pp.analyze(pp.build_origin_data(train, registry, specs, t0), circuits=circuits)

    audit = run_audit(train, registry, specs, t0, boundary, origins, tables, before)
    audit.append(("deterministic_repeat", digest(res) == digest(res_repeat),
                  f"two independent end-to-end runs give digest {digest(res)[:16]}"))
    after = tree_hashes(RESULTS, out)
    audit.append(("previous_result_files_unchanged", before == after, f"{len(before)} existing result files hashed before and after"))
    inputs_after = file_hashes([DEFAULT_CONFIG.default_dataset_path, DEFAULT_CONFIG.compatibility_registry_path])
    audit.append(("dataset_and_registry_unchanged", inputs_before == inputs_after, "sha256 of dataset.txt and compatibility_registry.csv unchanged"))

    write_csv(out / "integrity_audit.csv", ["check", "passed", "detail"], audit)
    if not all(a[1] for a in audit):
        (out / "INVALID.txt").write_text("An integrity check failed; no scientific metrics were published. See integrity_audit.csv.\n")
        print("INTEGRITY FAILURE:", [a[0] for a in audit if not a[1]], file=sys.stderr)
        return 2

    prim = res["outcomes"]["abs_d"]
    decision = pp.evaluate_decision(prim["priority"])

    write_csv(out / "origin_design.csv",
              ["origin", "window_index", "origin_time_s", "future_start_s", "future_end_s", "past_windows", "past_shots",
               "future_windows", "future_transitions_per_pair", "future_shots", "pairs_included", "pairs_excluded", "min_past_valid_transitions"],
              [[o.spec.index, o.spec.window_index, o.spec.origin_time, o.spec.origin_time, o.spec.future_end, o.spec.window_index,
                o.past_shots, o.spec.horizon_windows, o.spec.horizon_windows - 1, o.future_shots, int(o.included.sum()),
                int((~o.included).sum()), int(np.nanmin(o.past["past_valid"]))] for o in origins])
    rows = []
    for o in origins:
        fa, fz = o.outcome("abs_d"), o.outcome("z_sq")
        for i, (a, b) in enumerate(o.pairs):
            rows.append([o.spec.index, a, b, pp.wait_length(a), pp.wait_length(b), abs(pp.wait_length(a) - pp.wait_length(b)),
                         o.past["priority"][i], int(o.past["rank"][i]), o.past["mean_abs_z"][i], o.past["mean_abs_d"][i],
                         int(o.past["past_valid"][i]), fa[i], fz[i], int(o.n_valid[i]), bool(o.included[i]),
                         "" if o.included[i] else f"fewer than {pp.MIN_FUTURE_TRANSITIONS} valid future transitions"])
    write_csv(out / "pair_metrics_by_origin.csv",
              ["origin", "circuit_a", "circuit_b", "wait_length_a", "wait_length_b", "wait_length_gap", "priority_score",
               "priority_rank", "past_mean_abs_z", "past_mean_abs_d", "past_valid_transitions", "future_mean_abs_d",
               "future_mean_z_squared", "future_valid_transitions", "included", "exclusion_reason"], rows)

    corr, boot, perm, base = [], [], [], []
    n_pairs = [int(o.included.sum()) for o in origins]
    n_trans = [int(np.isfinite(o.abs_d[o.included]).sum()) for o in origins]
    for oc, label in (("abs_d", "future_mean_abs_d (primary)"), ("z_sq", "future_mean_z_squared (secondary)")):
        for n, r in res["outcomes"][oc].items():
            for i in range(len(origins)):
                corr.append([label, n, f"origin_{i}", r["sign"], n_pairs[i], n_trans[i], r["per_origin"][i]])
                boot.append([label, n, f"origin_{i}", r["per_origin"][i], r["per_origin_ci"][i][0], r["per_origin_ci"][i][1]])
                perm.append([label, n, f"origin_{i}", r["per_origin"][i], r["per_origin_perm_p"][i]])
            corr.append([label, n, "mean_of_origins_T", r["sign"], sum(n_pairs), sum(n_trans), r["T"]])
            corr.append([label, n, "pooled_descriptive_only", r["sign"], sum(n_pairs), sum(n_trans), r["pooled_descriptive"]])
            boot.append([label, n, "mean_of_origins_T", r["T"], r["T_ci"][0], r["T_ci"][1]])
            nt = r["null_T"]
            perm.append([label, n, "mean_of_origins_T", r["T"], r["T_perm_p"], nt["n"], nt["mean"], nt["sd"], nt["q05"], nt["q95"]])
            base.append([label, n, r["sign"], r["T"], r["sign"] * r["T"], r["T_ci"][0], r["T_ci"][1], r["T_perm_p"],
                         r["origins_in_expected_direction"], r["origins_defined"],
                         r["delta_vs_priority"]["estimate"], r["delta_vs_priority"]["ci"][0], r["delta_vs_priority"]["ci"][1]])
    write_csv(out / "priority_correlations.csv",
              ["outcome", "predictor", "scope", "expected_sign", "pairs", "transitions", "spearman"], corr)
    write_csv(out / "bootstrap_intervals.csv", ["outcome", "predictor", "scope", "estimate", "ci95_low", "ci95_high"], boot)
    perm = [r + [""] * (10 - len(r)) for r in perm]
    write_csv(out / "permutation_null.csv",
              ["outcome", "predictor", "scope", "observed", "p_one_sided_expected_direction", "null_n", "null_mean", "null_sd", "null_q05", "null_q95"], perm)
    write_csv(out / "baseline_comparison.csv",
              ["outcome", "predictor", "expected_sign", "T_raw", "T_aligned", "ci95_low", "ci95_high", "permutation_p",
               "origins_in_expected_direction", "origins_defined", "delta_aligned_vs_priority", "delta_ci95_low", "delta_ci95_high"], base)

    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if not k.startswith("_")}
        if isinstance(x, (list, tuple)):
            return [strip(v) for v in x]
        if isinstance(x, (np.floating, float)):
            return float(x) if math.isfinite(x) else None
        if isinstance(x, np.integer):
            return int(x)
        return x
    summary = {
        "verdict": decision["verdict"], "criteria": decision["criteria"],
        "design": {"window_seconds": pp.WINDOW_SECONDS, "origins": [dataclasses.asdict(s) for s in specs],
                   "min_future_transitions": pp.MIN_FUTURE_TRANSITIONS, "bootstrap_draws": pp.BOOTSTRAP_DRAWS,
                   "block_length": pp.BLOCK_LENGTH, "permutations": pp.PERMUTATIONS, "seed": pp.SEED,
                   "predictors_expected_sign": pp.PREDICTORS},
        "data": {"training_shots": len(train), "holdout_rows_discarded_unread": n_dropped, "boundary": boundary,
                 "eligible_pairs": 29, "eligible_circuits": len(circuits)},
        "results": strip(res), "integrity": {"checks": len(audit), "all_passed": True, "digest": digest(res)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    make_plots(out, res, origins)
    build_report(out, res, decision, origins, specs, audit, "")
    print(f"verdict: {decision['verdict']} ; T={prim['priority']['T']:+.4f} CI={prim['priority']['T_ci']} perm_p={prim['priority']['T_perm_p']:.4f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
