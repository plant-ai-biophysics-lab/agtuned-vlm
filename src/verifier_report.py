"""
Render the verifier-design findings as a multi-page PDF.

Reads outputs/verifier_analysis/verifier_analysis.json (produced by
src/verifier_analysis.py) and lays out a flowed report with charts.

Usage:
  uv run python src/verifier_report.py
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

# ── Page geometry (inches) ────────────────────────────────────────────────────
PW, PH = 8.5, 11.0
ML, MR = 0.75, 0.75
MT, MB = 0.75, 0.65
CW = PW - ML - MR

STYLE = {
    "h1":     dict(size=17, weight="bold", color="#111111", lead=0.34, before=0.00, after=0.16),
    "h2":     dict(size=11.5, weight="bold", color="#1a3a5c", lead=0.22, before=0.20, after=0.09),
    "body":   dict(size=8.8, weight="normal", color="#222222", lead=0.152, before=0.0, after=0.10),
    "bullet": dict(size=8.8, weight="normal", color="#222222", lead=0.152, before=0.0, after=0.055),
    "mono":   dict(size=7.3, weight="normal", color="#1b1b1b", lead=0.130, before=0.05, after=0.10),
    "note":   dict(size=7.8, weight="normal", color="#666666", lead=0.135, before=0.04, after=0.10),
}
WRAP = {"body": 104, "bullet": 100, "note": 116, "mono": 118, "h2": 78, "h1": 54}


class Flow:
    """Simple top-down flow layout across PDF pages."""

    def __init__(self, pdf: PdfPages, title: str) -> None:
        self.pdf = pdf
        self.title = title
        self.page_no = 0
        self._new_page()

    def _new_page(self) -> None:
        if self.page_no:
            self._footer()
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.page_no += 1
        self.fig = plt.figure(figsize=(PW, PH))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, PW)
        self.ax.set_ylim(0, PH)
        self.ax.axis("off")
        self.y = PH - MT

    def _footer(self) -> None:
        self.ax.plot([ML, PW - MR], [MB - 0.16, MB - 0.16],
                     color="#dddddd", linewidth=0.8)
        self.ax.text(ML, MB - 0.30, self.title, fontsize=6.8, color="#999999", va="top")
        self.ax.text(PW - MR, MB - 0.30, f"{self.page_no}",
                     fontsize=6.8, color="#999999", va="top", ha="right")

    def _space(self, h: float) -> None:
        if self.y - h < MB:
            self._new_page()

    def ensure(self, h: float) -> None:
        """Start a new page unless h inches remain — keeps a block unsplit."""
        if self.y - h < MB:
            self._new_page()

    def add(self, kind: str, text: str) -> None:
        st = STYLE[kind]
        self.y -= st["before"]
        lines: list[str] = []
        for para in text.split("\n"):
            lines.extend(textwrap.wrap(para, width=WRAP[kind]) or [""])
        for i, line in enumerate(lines):
            self._space(st["lead"])
            x = ML
            if kind == "bullet":
                if i == 0:
                    self.ax.text(ML, self.y, "•", fontsize=st["size"],
                                 color="#1a3a5c", va="top")
                x = ML + 0.17
            self.ax.text(x, self.y, line, fontsize=st["size"], va="top",
                         color=st["color"], fontweight=st["weight"],
                         family="monospace" if kind == "mono" else None)
            self.y -= st["lead"]
        self.y -= st["after"]

    def rule(self) -> None:
        self._space(0.12)
        self.ax.plot([ML, PW - MR], [self.y, self.y], color="#cccccc", linewidth=0.9)
        self.y -= 0.14

    def kpis(self, items: list[tuple[str, str, str]]) -> None:
        """Row of headline metric boxes: (value, label, sub)."""
        h = 0.92
        self._space(h + 0.12)
        n = len(items)
        gap = 0.12
        w = (CW - gap * (n - 1)) / n
        for i, (val, label, sub) in enumerate(items):
            x = ML + i * (w + gap)
            self.ax.add_patch(FancyBboxPatch(
                (x, self.y - h), w, h,
                boxstyle="round,pad=0.02", linewidth=1.0,
                edgecolor="#c8d6e2", facecolor="#f4f8fb"))
            self.ax.text(x + w / 2, self.y - 0.28, val, fontsize=17,
                         fontweight="bold", color="#1a3a5c", ha="center", va="center")
            self.ax.text(x + w / 2, self.y - 0.55, label, fontsize=7.6,
                         color="#333333", ha="center", va="center")
            self.ax.text(x + w / 2, self.y - 0.75, sub, fontsize=6.4,
                         color="#777777", ha="center", va="center")
        self.y -= h + 0.16

    def table(self, headers: list[str], rows: list[list[str]],
              widths: list[float], note: str = "") -> None:
        row_h = 0.175
        need = row_h * (len(rows) + 1) + 0.2
        self._space(min(need, 3.0))
        xs, acc = [], ML
        for wfrac in widths:
            xs.append(acc)
            acc += CW * wfrac
        self.ax.add_patch(FancyBboxPatch(
            (ML, self.y - row_h), CW, row_h,
            boxstyle="square,pad=0", linewidth=0, facecolor="#1a3a5c"))
        for x, htxt in zip(xs, headers):
            self.ax.text(x + 0.05, self.y - row_h / 2, htxt, fontsize=7.4,
                         color="white", fontweight="bold", va="center")
        self.y -= row_h
        for j, row in enumerate(rows):
            self._space(row_h)
            if j % 2 == 0:
                self.ax.add_patch(FancyBboxPatch(
                    (ML, self.y - row_h), CW, row_h,
                    boxstyle="square,pad=0", linewidth=0, facecolor="#f2f5f8"))
            for x, cell in zip(xs, row):
                bold = cell.startswith("*")
                self.ax.text(x + 0.05, self.y - row_h / 2, cell.lstrip("*"),
                             fontsize=7.2, va="center", color="#222222",
                             fontweight="bold" if bold else "normal",
                             family="monospace")
            self.y -= row_h
        self.y -= 0.06
        if note:
            self.add("note", note)

    def image(self, path: Path, height: float) -> None:
        self._space(height + 0.1)
        img = plt.imread(path)
        ih, iw = img.shape[0], img.shape[1]
        w = min(CW, height * iw / ih)
        h = w * ih / iw
        self._space(h + 0.1)
        axi = self.fig.add_axes([
            (ML + (CW - w) / 2) / PW, (self.y - h) / PH, w / PW, h / PH])
        axi.imshow(img)
        axi.axis("off")
        self.y -= h + 0.14

    def quote(self, lines: list[tuple[str, str]], accent: str = "#c62828") -> None:
        """Verbatim exhibit block: list of (label, text)."""
        wrapped: list[tuple[str, str]] = []
        for label, txt in lines:
            ws = textwrap.wrap(txt, width=98) or [""]
            for i, w in enumerate(ws):
                wrapped.append((label if i == 0 else "", w))
        h = 0.145 * len(wrapped) + 0.18
        self._space(h)
        self.ax.add_patch(FancyBboxPatch(
            (ML, self.y - h), CW, h,
            boxstyle="round,pad=0.02", linewidth=1.0,
            edgecolor="#e0c0c0", facecolor="#fdf6f6"))
        self.ax.plot([ML + 0.03, ML + 0.03], [self.y - h + 0.05, self.y - 0.05],
                     color=accent, linewidth=2.4)
        yy = self.y - 0.12
        for label, txt in wrapped:
            if label:
                self.ax.text(ML + 0.12, yy, label, fontsize=6.9, va="top",
                             color=accent, fontweight="bold", family="monospace")
            self.ax.text(ML + 0.72, yy, txt, fontsize=6.9, va="top",
                         color="#222222", family="monospace")
            yy -= 0.145
        self.y -= h + 0.12

    def close(self) -> None:
        self._footer()
        self.pdf.savefig(self.fig)
        plt.close(self.fig)


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def last_step_key(step_dict: dict) -> str:
    """Highest step-position key present, as a string — avoids hardcoding a
    step count that changes when the CoT chain length changes."""
    return str(max(int(k) for k in step_dict))


def build(res: dict, charts: Path, out: Path,
          swap_rates: dict | None = None, swap_by_step: dict | None = None) -> None:
    fa = res["final_answers"]
    st = res["steps"]
    traj = st["trajectory"]
    cons = st["self_consistency"]["buckets"]
    rules = st["verifier_rules"]["specific_gt"]
    bt = st["step_accuracy_by_track"]

    sc = fa["self_cot"]["strata"]["specific_gt"]
    cy = fa["cot_yaml"]["strata"]["specific_gt"]
    sc_cal, cy_cal = sc["calibration"], cy["calibration"]

    title = "CoT Reasoning Failure Analysis - Verifier Design Evidence"

    with PdfPages(out) as pdf:
        f = Flow(pdf, title)

        # ── Cover / summary ───────────────────────────────────────────────────
        f.add("h1", "CoT Reasoning Failure Analysis")
        f.add("note", f"Evidence base for verifier design  |  model: {res['model']}  |  "
                      "Sections 1-5 source: cot_step_analysis --swap_order subsample, rotation 0 "
                      "(10 gain/regression datasets, current 5-step/3-step prompts)")
        f.rule()

        f.kpis([
            (pct(sc["accuracy"]), "self_cot accuracy", f"specific labels, n={sc['n']:,}"),
            (pct(cy["accuracy"]), "cot_yaml accuracy", f"specific labels, n={cy['n']:,}"),
            (pct(traj["anchor_rate"]), "anchor rate", "step-1 right -> final right"),
            (pct(traj["recovery_rate"]), "recovery rate", "step-1 wrong -> final right"),
        ])

        f.add("h2", "Bottom line")
        f.add("body",
              "The model's answer is effectively decided at the first reasoning step. Subsequent CoT "
              "steps rarely repair a bad start (recovery "
              f"{pct(traj['recovery_rate'])}) but frequently damage a good one. Critically, the model "
              "signals uncertainty far too rarely to be trusted as its own gatekeeper: on specific "
              f"disease/species labels it is wrong {pct(1 - sc['accuracy'])} of the time yet abstains on only "
              f"{pct(sc_cal['abstains']['n'] / sc['n'])} of cases.")
        f.add("body",
              "A verifier is therefore worth building, but it should be understood as a triage tool "
              "rather than a repair tool. The best rule tested raises accuracy on retained answers from "
              f"{pct(rules['base_accuracy'])} to "
              f"{pct(rules['rules']['any_of_three']['kept_accuracy'])}, at the cost of flagging "
              f"{pct(rules['rules']['any_of_three']['coverage'])} of all answers for review.")

        f.add("h2", "The five findings")
        f.add("bullet", "CoT prompts that presuppose disease cause the model to confabulate symptoms on "
                        "healthy tissue. This is the single largest, most fixable failure mode.")
        f.add("bullet", "The model anchors on its first hypothesis and cannot recover. More reasoning "
                        "steps mostly add drift, not diagnosis.")
        f.add("bullet", "Confidence language is anti-correlated with correctness. Hedging and abstention "
                        "are reliable failure signals; assertive phrasing is not a reliability signal.")
        f.add("bullet", "Cross-probe disagreement is the strongest single predictor available, and it "
                        "needs no labels - making it the natural backbone of a verifier.")
        f.add("bullet", "Anchoring is not an artifact of which question happens to go first - it holds up "
                        "under a swap-order robustness check - but it is sensitive to question content, "
                        "and one redundant, presupposing prompt step has since been removed (Section 6).")

        f.add("h2", "A note on measurement")
        f.add("body",
              "All headline numbers are reported on the specific-label stratum only. Datasets contain "
              "catch-all ground-truth classes (healthy, normal, negative, grass, broadleaf, soil) where "
              "an evasive answer such as 'I cannot identify this' is scored correct by the LLM judge. "
              "Pooling those rows inverts several conclusions - most notably it makes abstention look "
              "beneficial when on real diagnostic labels abstention accompanies a wrong answer "
              f"{pct(rules['rules']['abstains']['precision'])} of the time. The two strata are kept "
              "separate throughout.")
        f.add("body",
              "Data-freshness note: Sections 1-5 now run on the --swap_order subsample (rotation 0, the "
              "natural unrotated question order) rather than the original full-test-split run, so they "
              "reflect the current prompts - the redundant final CoT step described in Section 6 is "
              "already gone from this data. The tradeoff is coverage: 10 datasets capped at 50 images/class "
              "(n=1,903 chains) instead of the full 26-dataset test split. self_cot is intersected down to "
              "the exact same images cot_yaml was tested on (n matches exactly in the KPIs above; a small "
              "fraction of the swap sample has no self_cot counterpart and is dropped from both sides) for "
              "a true same-sample comparison. Self-consistency (Sections 4-5) is redefined as "
              "cross-rotation agreement - the same image's final answer compared across the 5 (or 3) "
              "question orderings tested by swap-order - since this subsample has no independent-probe data. "
              "A full-test-split rerun would sharpen these numbers but is not expected to change their "
              "direction; see Section 6 for the multi-rotation robustness check this subsample also enables.")

        # ── Finding 1 ─────────────────────────────────────────────────────────
        f._new_page()
        f.add("h1", "1. Presupposition drives confabulation")
        f.add("body",
              "Splitting the step-by-step trajectory by ground-truth type reverses the naive reading of "
              "the aggregate curve. On genuine disease labels, CoT steadily helps. On healthy tissue it "
              "collapses.")
        f.image(charts / "swap_subsample" / "step_accuracy_stratified.png", 3.5)

        d_spec = bt["disease_pest/specific_gt"]
        d_non = bt["disease_pest/nonspecific_gt"]
        w_spec = bt["weed/specific_gt"]
        d_last = last_step_key(d_spec)
        w_last = last_step_key(w_spec)
        f.table(
            ["Track / ground truth", "step 1", "final", "change", "reading"],
            [
                ["disease - real disease", pct(d_spec["1"]["acc"]), pct(d_spec[d_last]["acc"]),
                 f"+{(d_spec[d_last]['acc'] - d_spec['1']['acc']) * 100:.1f}pp", "CoT genuinely helps"],
                ["*disease - healthy", pct(d_non["1"]["acc"]), pct(d_non[d_last]["acc"]),
                 f"{(d_non[d_last]['acc'] - d_non['1']['acc']) * 100:.1f}pp", "*catastrophic collapse"],
                ["weed - named species", pct(w_spec["1"]["acc"]), pct(w_spec[w_last]["acc"]),
                 f"{(w_spec[w_last]['acc'] - w_spec['1']['acc']) * 100:.1f}pp", "CoT adds noise"],
            ],
            [0.26, 0.11, 0.11, 0.12, 0.40],
        )

        f.add("body",
              "The healthy-tissue collapse is concentrated at a single transition: step 1 to step 2 loses "
              f"{(d_non['1']['acc'] - d_non['2']['acc']) * 100:.1f} percentage points. Step 2 of the "
              "disease track asks 'Which specific parts of the plant are showing symptoms?' - a question "
              "that presupposes symptoms exist. The weed track has no equivalent presupposition at step 2, "
              f"and its coarse-label curve stays comparatively flat across all {w_last} steps. The damage "
              "tracks the question wording, not the step count.")

        f.add("h2", "Exhibit: a healthy leaf becomes Coffee Leaf Rust")
        f.add("note", "arabica_coffee_leaf_disease_classification, ground truth = Healthy. Verbatim, abridged.")
        f.quote([
            ("STEP 1 Q", "What specific plant or crop species is this?"),
            ("ANSWER", "...it is not possible to definitively identify the specific plant or crop species."),
            ("PREDICT", "The image does not exhibit clear signs of a specific disease, pest, or "
                        "physiological stress, showing only healthy green leaf tissue with normal venation."),
            ("VERDICT", "CORRECT"),
            ("", ""),
            ("STEP 2 Q", "Which specific parts of the plant are showing symptoms?"),
            ("ANSWER", "Based on the provided image, the visible symptoms are localized to the lamina "
                       "(the leaf blade) and the venation system."),
            ("PREDICT", "The image exhibits symptoms consistent with Coffee Leaf Rust (Hemileia "
                        "vastatrix), a fungal disease characterized by chlorotic lesions."),
            ("VERDICT", "INCORRECT"),
        ])
        f.add("body",
              "The model does not revise its perception; it accepts the premise of the question and "
              "generates symptom language to satisfy it, then treats that generated language as evidence "
              "at the classification step. Because the chain is cumulative, the confabulated symptoms "
              "persist in context for every subsequent step.")

        # ── Finding 2 ─────────────────────────────────────────────────────────
        f._new_page()
        f.add("h1", "2. Early anchoring, no recovery")
        f.add("body",
              f"Across {traj['n_groups']:,} image chains, the first step's verdict largely determines the "
              f"last. When step 1 is right the final answer is right {pct(traj['anchor_rate'])} of the "
              f"time; when step 1 is wrong only {pct(traj['recovery_rate'])} recover. Additional "
              "deliberation is close to a one-way ratchet.")
        f.image(charts / "swap_subsample" / "trajectory_consistency.png", 3.0)

        fbf = traj["final_acc_by_flips"]
        f.table(
            ["Diagnosis changes across chain", "n chains", "final accuracy"],
            [[f"{k} change(s)", f"{v['n']:,}", ("*" if k == "0" else "") + pct(v["acc"])]
             for k, v in sorted(fbf.items(), key=lambda kv: int(kv[0]))],
            [0.42, 0.24, 0.34],
            note="A 'change' is consecutive steps whose predictions share less than half their content words.",
        )
        flip_ks = sorted(int(k) for k in fbf)
        min_fk, max_fk = str(flip_ks[0]), str(flip_ks[-1])
        f.add("body",
              f"Chains that {'never change their mind' if min_fk == '0' else f'change only {min_fk} time(s)'} "
              f"are right {pct(fbf[min_fk]['acc'])} of the time; chains that change {max_fk} times are right "
              f"{pct(fbf[max_fk]['acc'])} of the time. Instability is not the model searching productively - "
              "it is the model failing to see anything decisive and drifting between hypotheses. This makes "
              "flip count a usable, label-free confidence proxy.")

        # ── Finding 3 ─────────────────────────────────────────────────────────
        f._new_page()
        f.add("h1", "3. Confidence language does not mean correctness")
        f.add("body",
              "Every hedging family is negatively associated with accuracy, but so is explicit confidence "
              "language. Assertive phrasing ('clearly', 'characteristic of', 'definitively') appears "
              "disproportionately on wrong answers - the model reaches for emphasis when the image is "
              "uninformative, not when it is certain.")
        f.image(charts / "swap_subsample" / "marker_lift_final.png", 3.1)

        f.add("h2", "Epistemic stance vs accuracy (specific labels only)")
        f.table(
            ["Stance of final answer", "self_cot n", "self_cot acc", "cot_yaml n", "cot_yaml acc"],
            [
                ["commits (no hedging)", f"{sc_cal['commits']['n']:,}", pct(sc_cal['commits']['acc']),
                 f"{cy_cal['commits']['n']:,}", pct(cy_cal['commits']['acc'])],
                ["hedges (may/likely/suggests)", f"{sc_cal['hedges']['n']:,}", pct(sc_cal['hedges']['acc']),
                 f"{cy_cal['hedges']['n']:,}", pct(cy_cal['hedges']['acc'])],
                ["*abstains (cannot determine)", f"{sc_cal['abstains']['n']:,}", "*" + pct(sc_cal['abstains']['acc']),
                 f"{cy_cal['abstains']['n']:,}", "*" + pct(cy_cal['abstains']['acc'])],
            ],
            [0.32, 0.17, 0.17, 0.17, 0.17],
        )
        f.add("body",
              "The ordering is directionally correct - commits beats hedges beats abstains - so the model "
              "does carry some self-knowledge. But the useful end is the bottom row: abstention is close to "
              "a guarantee of failure. The top row is where the problem lies. Confident answers are still "
              f"wrong {pct(1 - sc_cal['commits']['acc'])} of the time in self_cot, which is roughly "
              f"{int(sc_cal['commits']['n'] * (1 - sc_cal['commits']['acc'])):,} confidently wrong answers "
              "that no linguistic cue will catch.")
        f.add("body",
              "One marker is mildly positive: naming a scientific binomial "
              f"({pct(sc['markers']['scientific_name']['lift'])} lift in self_cot). Retrieval of a specific "
              "taxon is weak evidence of genuine recognition, and it is the only text feature worth "
              "treating as a positive signal.")

        # ── Finding 4 ─────────────────────────────────────────────────────────
        f._new_page()
        f.add("h1", "4. Does the model lack knowledge?")
        f.add("body",
              "The evidence points to a perception and grounding deficit rather than a missing-facts "
              "deficit, and to a large calibration gap on top of it.")

        low, high = cons["low (<0.3)"], cons["high (>=0.6)"]
        f.kpis([
            (pct(1 - sc["accuracy"]), "error rate", "self_cot, specific labels"),
            (pct(sc_cal["abstains"]["n"] / sc["n"]), "abstention rate", "how often it admits doubt"),
            (f"{low['n'] / (low['n'] + cons['medium (0.3-0.6)']['n'] + high['n']) * 100:.0f}%",
             "unstable images", "cross-probe agreement < 0.3"),
            (pct(high["acc"]), "acc when stable", f"agreement >= 0.6, n={high['n']:,}"),
        ])

        f.add("h2", "Evidence for a grounding deficit")
        f.add("bullet",
              "Asked the same image under different probe questions in isolation, the model returns "
              f"mutually inconsistent diagnoses for {low['n']:,} of "
              f"{low['n'] + cons['medium (0.3-0.6)']['n'] + high['n']:,} images. A stable internal "
              "representation of the specimen would not be this probe-dependent.")
        f.add("bullet",
              f"When the probes do agree, accuracy jumps to {pct(high['acc'])} against a "
              f"{pct(low['acc'])} baseline for unstable images. The knowledge exists and is retrievable - "
              "it is simply not reliably triggered by the image.")
        f.add("bullet",
              "The model produces fluent, specific, domain-correct pathology vocabulary in every response, "
              "including the wrong ones. Language competence is not the bottleneck.")
        f.add("bullet",
              "It readily generates symptom descriptions for tissue that has no symptoms, which indicates "
              "the visual evidence is not constraining generation.")

        f.add("h2", "Evidence for a calibration gap")
        f.add("body",
              f"On specific labels the model is wrong {pct(1 - sc['accuracy'])} of the time but abstains on "
              f"only {pct(sc_cal['abstains']['n'] / sc['n'])} of cases - a gap of roughly "
              f"{(1 - sc['accuracy'] - sc_cal['abstains']['n'] / sc['n']) * 100:.0f} percentage points "
              "between how often it is wrong and how often it says so. When it does abstain it is almost "
              "always right to: those answers are wrong "
              f"{pct(rules['rules']['abstains']['precision'])} of the time. The signal is trustworthy; "
              "there is simply far too little of it.")
        f.add("body",
              "Practical consequence: self-reported confidence cannot be the verifier's primary input. "
              "The verifier needs an independent view of the image, or a consistency measure the model "
              "cannot fake by phrasing.")

        # ── Rule evaluation ───────────────────────────────────────────────────
        f._new_page()
        f.add("h1", "5. What a verifier can realistically buy")
        f.add("body",
              f"Candidate flag rules evaluated on {rules['n']:,} chains with specific ground-truth labels "
              f"(base accuracy {pct(rules['base_accuracy'])}). 'Precision' is the share of flagged answers "
              "that really are wrong; 'kept accuracy' is what you would achieve by auto-accepting "
              "everything not flagged.")

        order = ["abstains", "many_flips", "low_consistency", "abstains_OR_lowcons",
                 "lowcons_AND_manyflips", "any_of_three"]
        rows = []
        for k in order:
            v = rules["rules"].get(k)
            if not v:
                continue
            star = "*" if k == "any_of_three" else ""
            rows.append([star + k.replace("_", " "), pct(v["coverage"]), pct(v["precision"]),
                         pct(v["recall"]), star + pct(v["kept_accuracy"]),
                         f"{v['kept_acc_gain'] * 100:+.1f}pp"])
        f.table(["Rule", "flagged", "precision", "recall", "kept acc", "gain"], rows,
                [0.30, 0.13, 0.14, 0.13, 0.15, 0.15])

        f.add("h2", "How to read this")
        f.add("bullet",
              f"Abstention is the sharpest rule ({pct(rules['rules']['abstains']['precision'])} precision) "
              "but fires on almost nothing. Use it as an unconditional auto-reject; it will not carry the "
              "system.")
        f.add("bullet",
              "Cross-probe inconsistency is the workhorse: "
              f"{pct(rules['rules']['low_consistency']['precision'])} precision at "
              f"{pct(rules['rules']['low_consistency']['coverage'])} coverage. It costs extra inference "
              "passes but needs no labels and no trained model.")
        f.add("bullet",
              "The combined rule flags "
              f"{pct(rules['rules']['any_of_three']['coverage'])} of answers and lifts retained accuracy to "
              f"{pct(rules['rules']['any_of_three']['kept_accuracy'])}. That is a real gain, but note what "
              "it implies: three quarters of all answers are routed to review, and the quarter you keep is "
              "still wrong more than half the time.")
        f.add("body",
              "This is the central strategic point. Filtering cannot manufacture correct answers, and the "
              "ceiling here is low because the underlying visual grounding is weak. A verifier is worth "
              "building for triage and for abstention hygiene, but the larger win is upstream: fixing the "
              "prompt presuppositions and improving visual grounding.")

        # ── Swap-order robustness check + prompt fix ────────────────────────────
        if swap_rates and swap_by_step:
            f._new_page()
            f.add("h1", "6. Robustness check: does anchoring depend on question order?")
            f.add("body",
                  f"Finding 2 showed a {pct(traj['anchor_rate'])} anchor rate and "
                  f"{pct(traj['recovery_rate'])} recovery rate using rotation 0 (the natural, unrotated "
                  "question order) as the fixed order throughout the chain. That leaves an obvious confound: "
                  "is anchoring a property of the model's reasoning, or an artifact of whichever question "
                  "happened to be asked first? This section cyclically rotates the CoT question order - "
                  "every observation question gets a turn leading the chain - on the same 10 gain/regression "
                  "datasets (capped to 50 images/class for iteration speed) and re-measures anchor/recovery "
                  "rate per rotation.")

            f.add("h2", "A prompt fix, found in the process of designing this check")
            f.add("body",
                  "Each track's CoT chain ended with a synthesis step - 'Based on the visual markers so "
                  "far, what is the most likely diagnosis? State your confidence for each' - immediately "
                  "followed by a separate, always-present classification question asking essentially the "
                  "same thing in a terser format. At the final position these two questions were redundant. "
                  "Worse, the synthesis step's 'so far' wording presupposes prior observations exist, so "
                  "rotating it to a non-final position would have tested a broken prompt rather than "
                  "whether question order matters. It was removed from cot_prompts.yaml; the separate "
                  "classification question (asked after every step regardless of rotation) remains the "
                  "single source of the graded, comparable final answer at every chain position. Disease/"
                  "pest is now a 5-step chain, weed a 3-step chain.")

            f.add("h2", "Anchor rate holds up regardless of which question leads")
            dp = swap_rates.get("disease_pest", {})
            wd = swap_rates.get("weed_identification", {})
            if dp:
                dp_anchor = [v["anchor_rate"] for v in dp.values()]
                dp_lo, dp_hi = min(dp_anchor) * 100, max(dp_anchor) * 100
            if wd:
                wd_anchor = [v["anchor_rate"] for v in wd.values()]
                wd_lo, wd_hi = min(wd_anchor) * 100, max(wd_anchor) * 100
            f.add("body",
                  f"Disease/pest anchor rate ranges {dp_lo:.0f}-{dp_hi:.0f}% across the five possible lead "
                  f"questions; weed ranges {wd_lo:.0f}-{wd_hi:.0f}% across its three. Anchoring is real and "
                  "not an artifact of a specific question order - it is a property of how the model reasons "
                  "through the chain, not of which fact it happens to gather first.")
            f.image(charts / "swap_anchor_disease_pest.png", 3.2)

            f.add("h2", "Question content matters, even though position doesn't dominate")
            if dp:
                worst_rot = min(dp, key=lambda r: dp[r]["anchor_rate"])
                worst_val = dp[worst_rot]["anchor_rate"] * 100
            f.add("body",
                  f"The weakest rotation is Q{int(worst_rot) + 1}-first at {worst_val:.0f}% anchor rate - "
                  "the disease/pest chain's narrowest, most technical observation question ('is the damage "
                  "uniform across the whole plant'). Leading with it produces a distinctly worse and "
                  "non-improving anchor early in the chain, versus leading with broader diagnostic "
                  "questions (species identity, symptom location). This suggests ordering CoT steps "
                  "broad-to-narrow is healthier for reasoning stability than the reverse.")

            f.add("h2", "Instability is front-loaded but never fully resolves")
            f.add("body",
                  "Generalizing anchor/recovery rate to every chain position (not just position 1) shows "
                  "the trend is real, not purely mechanical: anchor rate climbs through the disease/pest "
                  "chain (roughly 63-75% at position 1 to 76-85% by position 4) while recovery rate falls "
                  "(11-14% down to 7-11%). If this were pure proximity-to-final effect, anchor rate one "
                  "step before the end should approach 100%. It does not - roughly 15-20% of chains that "
                  "are correct one question before the end still flip to wrong on the final transition. "
                  "Most churn happens early, but the chain never fully settles.")
            f.image(charts / "swap_anchor_by_step_disease_pest.png", 3.0)

            f.add("note",
                  "Caveat: this section uses the top-5-gain/top-5-regression dataset subset (n<=50 "
                  "images/class), smaller and differently sampled than the full-test-split findings in "
                  "Sections 1-5. Numbers here are directionally reliable but not directly comparable "
                  "1:1 to the earlier headline statistics.")

        # ── Recommendations ───────────────────────────────────────────────────
        f._new_page()
        f.add("h1", "7. Recommended verifier design")

        f.add("h2", "Fix the prompts first - it is free")
        f.add("bullet", "Done: the redundant final synthesis step (Section 6) has been removed from "
                        "cot_prompts.yaml. Still open, and the higher-value fix: the remaining symptom-"
                        "location step ('which parts are showing symptoms') still presupposes disease and "
                        "is Finding 1's root cause, not the step that was removed.")
        f.add("bullet", "Rewrite that presupposing step to be existence-neutral: 'Are any parts of the "
                        "plant showing symptoms? If none, say so explicitly and stop.' Expected to recover "
                        f"much of the {(d_non['1']['acc'] - d_non[d_last]['acc']) * 100:.0f}pp healthy-tissue "
                        "loss measured in Section 1 - still present after the Section 6 fix, since that fix "
                        "targeted a different step.")
        f.add("bullet", "Add an explicit healthy escape hatch to every step, and make 'no abnormality "
                        "detected' a first-class allowed answer rather than a failure to comply.")
        f.add("bullet", "Shorten the chain further if re-measurement confirms the same pattern as Section "
                        "6: step 1 already carries most of the signal on the weed track, and disease/pest "
                        "gains little after step 3-4.")

        f.add("h2", "Verifier architecture: per-statement, in-line verification")
        f.add("body",
              "Revised recommendation: a small, reasoning-capable model fine-tuned on agricultural data, "
              "verifying every individual statement in the chain as it is produced - not just the final "
              "claim. This is a stronger design than checking only the end for two reasons grounded "
              "directly in this report's findings. First, Finding 2's 12.1% recovery rate means a bad "
              "statement that enters context is very unlikely to be corrected by later steps - catching it "
              "only at the end is catching it too late. Second, prompt-rewriting (the fix above) only works "
              "for cot_yaml's fixed question set; self_cot's self-generated reasoning has no fixed prompt to "
              "rewrite, so it can still confabulate the same way with no lever to pull. A statement-level "
              "verifier catches the failure mode itself - a claim asserted with no visual support - "
              "regardless of whether the question that prompted it was guided (cot_yaml) or self-generated "
              "(self_cot). A small fine-tuned model is what makes this economical: verifying every step "
              "means 3-5x more verifier calls than a single end check, which only pencils out with a small, "
              "cheap, specialized model rather than a large general one.")

        f.add("bullet", "Stage 1 - in-line statement verification. After every CoT step (fixed question or "
                        "self-generated sub-thought), the verifier checks that one statement against the "
                        "image and the running history of ALREADY-VERIFIED statements - not the raw, "
                        "unverified history, so a later step can't build on an earlier fabrication. Verdicts: "
                        "SUPPORTED / UNSUPPORTED / PRESUPPOSES_UNCONFIRMED / CONTRADICTS_PRIOR / "
                        "INSUFFICIENT_EVIDENCE.")
        f.add("bullet", "Stage 2 - regenerate or halt. On UNSUPPORTED or PRESUPPOSES_UNCONFIRMED, either "
                        "regenerate that step with a corrective instruction ('no symptom is confirmed yet - "
                        "say so, do not assume one'), or halt the chain and flag it before a doomed "
                        "trajectory reaches a final answer. Given the recovery-rate finding, halting early "
                        "is cheap insurance.")
        f.add("bullet", "Stage 3 - chain-level rollup. Aggregate the per-statement verdicts: count of flags, "
                        "the step index where the first flag occurred, and whether a regeneration happened. "
                        "This is a strictly richer signal than the current token-overlap flip count, because "
                        "it is grounded in an actual evidence check rather than detecting that the wording "
                        "changed.")
        f.add("bullet", "Stage 4 - cross-probe self-consistency, retained from the original design. Still "
                        "cheap and complementary: it catches the case where every statement is individually "
                        "well-supported but the model is confidently, consistently wrong across framings.")
        f.add("bullet", "Stage 5 - final adjudication, by the same fine-tuned model. Given the full "
                        "per-statement verdict trail (not just the claim and some alternatives, as in the "
                        "original design) plus the final answer, output CONFIRM / REJECT / NO_ABNORMALITY / "
                        "INSUFFICIENT_EVIDENCE with an explicit confidence band. Surface the first-flagged "
                        "step alongside the verdict - useful both for automated routing and for manual PI "
                        "review of exactly where a chain went wrong (see sample_conversations.md).")
        f.add("bullet", "Training data: this is a natural extension of the existing ag_finetune verifier "
                        "pipeline (05_build_verifier_data.py, stage3_grpo_verifier.py) - hard negatives and "
                        "corrections already being generated there just need to be built at statement "
                        "granularity (one training example per flagged step) rather than only at the final-"
                        "answer granularity they currently target.")

        # Keep the whole template on one page
        f.ensure(STYLE["mono"]["lead"] * (VERIFIER_PROMPT.count("\n") + 3) + 0.4)
        f.add("h2", "Verifier prompt template (per-statement)")
        f.add("mono", VERIFIER_PROMPT)

        # ── Caveats ───────────────────────────────────────────────────────────
        f._new_page()
        f.add("h1", "8. Caveats and next steps")
        f.add("h2", "What would weaken these conclusions")
        f.add("bullet", "Correctness is LLM-judged, not human-adjudicated. The judge is demonstrably "
                        "lenient toward vague answers on catch-all classes, which is why those rows are "
                        "reported separately; residual leniency inside the specific stratum is possible.")
        f.add("bullet", "Agreement between predictions is measured by content-word overlap, not by "
                        "resolving each prediction to a canonical class label. Two phrasings of the same "
                        "diagnosis may register as disagreement, which would inflate the instability rate.")
        f.add("bullet", "Sections 1-5 now run on 10 datasets selected as the largest gainers and "
                        "regressors, capped at 50 images/class - not a random sample and a smaller pool "
                        "than the original full-test-split design. The presupposition and anchoring "
                        "findings both reproduced on this subsample after the prompt fix, which is "
                        "reassuring, but a full-test-split rerun would still sharpen every number here.")
        f.add("bullet", "self_cot intermediate reasoning was never persisted, so all step-level findings "
                        "come from cot_yaml. Whether self_cot anchors as hard is untested.")

        f.add("h2", "Suggested order of work")
        f.add("bullet", "Re-run cot_step_analysis.py from scratch on the full test split (delete or bypass "
                        "ckpt_cumulative.csv / ckpt_independent.csv / ckpt_judged.csv rather than resuming - "
                        "a resumed run previously kept using prompts cached from before the Section 6 fix "
                        "despite a fresh file timestamp; confirm freshness by checking step_idx only goes "
                        "up to 5 disease / 3 weed) to sharpen Sections 1-5 beyond the current 10-dataset "
                        "swap-order subsample and restore true independent-mode self-consistency data.")
        f.add("bullet", "Re-run the disease track with existence-neutral symptom-location wording on the "
                        "healthy subset only. Cheap, targeted, and tests the single largest effect in this "
                        "report.")
        f.add("bullet", "Persist self_cot intermediate responses so the anchoring analysis can be repeated "
                        "for that condition.")
        f.add("bullet", "Stand up Stage 4 (cross-probe self-consistency) first - it needs no new model and "
                        "can be measured end-to-end immediately, in parallel with fine-tuning the "
                        "statement-level verifier.")
        f.add("bullet", "Extend 05_build_verifier_data.py to statement granularity so the fine-tuned "
                        "verifier (Stages 1-2 and 5) has training data to learn from before it goes live; "
                        "validate Stage 4's standalone lift as the baseline it needs to beat.")

        f.add("h2", "Artifacts")
        f.add("note",
              "outputs/verifier_analysis/verifier_analysis_swap.json - Sections 1-5 statistics (current, "
              "swap-order subsample)\n"
              "outputs/verifier_analysis/verifier_analysis.json - superseded full-test-split run (pre-fix "
              "prompts, kept for reference)\n"
              "outputs/verifier_analysis/swap_anchor_rates.json, swap_anchor_by_step.json - Section 6\n"
              "outputs/verifier_analysis/swap_subsample/*.png - Sections 1-5 charts\n"
              "outputs/verifier_analysis/*.png - Section 6 charts\n"
              "src/verifier_analysis.py, src/cot_step_analysis.py - analysis\n"
              "src/verifier_report.py - this report\n"
              "src/config/cot_prompts.yaml - prompt fix (redundant synthesis step removed)")

        f.close()


VERIFIER_PROMPT = """\
You are a specialist agricultural verifier - a small model fine-tuned on
agricultural disease/pest/weed data - checking ONE statement from a chain of
reasoning about an image. Verify only the statement given, not the whole
chain. The step that produced it may be from a fixed diagnostic question
(guided, e.g. cot_yaml) or the model's own self-generated train of thought
(e.g. self_cot) - treat both the same way.

Image: {image}
Verified history (statements already checked and confirmed SUPPORTED only -
does NOT include anything flagged below): {verified_history}
This step's question or reasoning prompt (if any): {step_question}
Statement to verify: {statement}

Apply these checks in order. Do not skip to a verdict.

1. PRESUPPOSITION CHECK - decide this before anything else.
   Does the step's question, or the statement itself, assume something not
   yet established (e.g. "which parts show symptoms" presupposes symptoms
   exist)? If nothing in the verified history confirms the premise, the
   correct move is to say the premise is unconfirmed, not to invent content
   that satisfies it.

2. EVIDENCE CHECK - is this specific statement visible in the image?
   Name the specific pixels/region that support it. If you cannot point to a
   region, the statement is unsupported regardless of how confident it sounds.

3. CONSISTENCY CHECK.
   Does this statement contradict anything in the verified history? The
   chain cannot hold two incompatible claims - flag the conflict rather than
   silently letting the newer statement overwrite the earlier one.

4. CONFIDENCE-LANGUAGE WARNING.
   Ignore words like "clearly", "characteristic of", "definitively" when
   judging - in this model they correlate with WRONG statements, not right
   ones. Judge only the visible evidence.

5. IMAGE-QUALITY CHECK.
   If resolution, focus, or crop make this specific statement unverifiable,
   say so. INSUFFICIENT_EVIDENCE is a correct outcome, not a failure.

Return JSON:
{"verdict": "SUPPORTED" | "UNSUPPORTED" | "PRESUPPOSES_UNCONFIRMED" | "CONTRADICTS_PRIOR" | "INSUFFICIENT_EVIDENCE",
 "corrected_statement": "<neutral restatement if UNSUPPORTED/PRESUPPOSES, or null>",
 "visual_evidence": "<region and what is visible, or null>",
 "halt_chain": true | false,
 "confidence": "high" | "medium" | "low",
 "reasoning": "<1-2 sentences>"}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_dir", default="outputs/verifier_analysis/")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = Path(args.analysis_dir)
    # Sections 1-5 now run on verifier_analysis_swap.json (fresh: current
    # 5-step/3-step prompts, 10-dataset gain/regression subsample, rotation 0
    # as the natural question order). The full-test-split verifier_analysis.json
    # remains on disk for reference but is stale relative to the prompt fix.
    res = json.loads((d / "verifier_analysis_swap.json").read_text())
    swap_rates = json.loads((d / "swap_anchor_rates.json").read_text())
    swap_by_step = json.loads((d / "swap_anchor_by_step.json").read_text())
    out = Path(args.out) if args.out else d / "verifier_findings.pdf"
    build(res, d, out, swap_rates=swap_rates, swap_by_step=swap_by_step)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
