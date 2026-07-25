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


def build(res: dict, charts: Path, out: Path) -> None:
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
                      "sources: cot_vision (final answers), cot_step_analysis (intermediate steps)")
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

        f.add("h2", "The four findings")
        f.add("bullet", "CoT prompts that presuppose disease cause the model to confabulate symptoms on "
                        "healthy tissue. This is the single largest, most fixable failure mode.")
        f.add("bullet", "The model anchors on its first hypothesis and cannot recover. More reasoning "
                        "steps mostly add drift, not diagnosis.")
        f.add("bullet", "Confidence language is anti-correlated with correctness. Hedging and abstention "
                        "are reliable failure signals; assertive phrasing is not a reliability signal.")
        f.add("bullet", "Cross-probe disagreement is the strongest single predictor available, and it "
                        "needs no labels - making it the natural backbone of a verifier.")

        f.add("h2", "A note on measurement")
        f.add("body",
              "All headline numbers are reported on the specific-label stratum only. Datasets contain "
              "catch-all ground-truth classes (healthy, normal, negative, grass, broadleaf, soil) where "
              "an evasive answer such as 'I cannot identify this' is scored correct by the LLM judge. "
              "Pooling those rows inverts several conclusions - most notably it makes abstention look "
              "beneficial when on real diagnostic labels abstention accompanies a wrong answer "
              f"{pct(rules['rules']['abstains']['precision'])} of the time. The two strata are kept "
              "separate throughout.")

        # ── Finding 1 ─────────────────────────────────────────────────────────
        f._new_page()
        f.add("h1", "1. Presupposition drives confabulation")
        f.add("body",
              "Splitting the step-by-step trajectory by ground-truth type reverses the naive reading of "
              "the aggregate curve. On genuine disease labels, CoT steadily helps. On healthy tissue it "
              "collapses.")
        f.image(charts / "step_accuracy_stratified.png", 3.5)

        d_spec = bt["disease_pest/specific_gt"]
        d_non = bt["disease_pest/nonspecific_gt"]
        w_spec = bt["weed/specific_gt"]
        f.table(
            ["Track / ground truth", "step 1", "final", "change", "reading"],
            [
                ["disease - real disease", pct(d_spec["1"]["acc"]), pct(d_spec["6"]["acc"]),
                 f"+{(d_spec['6']['acc'] - d_spec['1']['acc']) * 100:.1f}pp", "CoT genuinely helps"],
                ["*disease - healthy", pct(d_non["1"]["acc"]), pct(d_non["6"]["acc"]),
                 f"{(d_non['6']['acc'] - d_non['1']['acc']) * 100:.1f}pp", "*catastrophic collapse"],
                ["weed - named species", pct(w_spec["1"]["acc"]), pct(w_spec["4"]["acc"]),
                 f"{(w_spec['4']['acc'] - w_spec['1']['acc']) * 100:.1f}pp", "CoT adds noise"],
            ],
            [0.26, 0.11, 0.11, 0.12, 0.40],
        )

        f.add("body",
              "The healthy-tissue collapse is concentrated at a single transition: step 1 to step 2 loses "
              f"{(d_non['1']['acc'] - d_non['2']['acc']) * 100:.1f} percentage points. Step 2 of the "
              "disease track asks 'Which specific parts of the plant are showing symptoms?' - a question "
              "that presupposes symptoms exist. The weed track has no equivalent presupposition at step 2, "
              "and its coarse-label curve stays flat across all four steps. The damage tracks the question "
              "wording, not the step count.")

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
        f.image(charts / "trajectory_consistency.png", 3.0)

        fbf = traj["final_acc_by_flips"]
        f.table(
            ["Diagnosis changes across chain", "n chains", "final accuracy"],
            [[f"{k} change(s)", f"{v['n']:,}", ("*" if k == "0" else "") + pct(v["acc"])]
             for k, v in sorted(fbf.items(), key=lambda kv: int(kv[0]))],
            [0.42, 0.24, 0.34],
            note="A 'change' is consecutive steps whose predictions share less than half their content words.",
        )
        f.add("body",
              "Chains that never change their mind are right "
              f"{pct(fbf['0']['acc'])} of the time; chains that change five times are right "
              f"{pct(fbf['5']['acc'])} of the time. Instability is not the model searching productively - "
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
        f.image(charts / "marker_lift_final.png", 3.1)

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

        # ── Recommendations ───────────────────────────────────────────────────
        f._new_page()
        f.add("h1", "6. Recommended verifier design")

        f.add("h2", "Fix the prompts first - it is free")
        f.add("bullet", "Rewrite presupposing steps to be existence-neutral: 'Are any parts of the plant "
                        "showing symptoms? If none, say so explicitly and stop.' Expected to recover much "
                        f"of the {(d_non['1']['acc'] - d_non['6']['acc']) * 100:.0f}pp healthy-tissue loss.")
        f.add("bullet", "Add an explicit healthy escape hatch to every step, and make 'no abnormality "
                        "detected' a first-class allowed answer rather than a failure to comply.")
        f.add("bullet", "Shorten the chain. Step 1 already carries most of the signal on the weed track, "
                        "and the disease track gains little after step 4.")

        f.add("h2", "Verifier architecture")
        f.add("body",
              "Use a general model with a structured rubric first. A fine-tuned verifier is only worth the "
              "cost once the prompt fixes are in and you have labelled disagreement cases to train on - "
              "and the analysis suggests its ceiling is set by visual grounding, not by judgement quality.")
        f.add("bullet", "Stage A - self-consistency probe. Re-ask the classification under 3 independent "
                        "framings, no shared context. Compute content-word agreement. This is the primary "
                        "signal and requires no verifier model at all.")
        f.add("bullet", "Stage B - rule gate. Auto-reject on abstention or hedge_strong. Auto-flag when "
                        "agreement < 0.3 or diagnosis changed 4+ times.")
        f.add("bullet", "Stage C - LLM verifier, invoked only on flagged cases. Give it the image, the "
                        "claim, and the disagreeing alternatives, and ask it to adjudicate against a rubric.")
        f.add("bullet", "Stage D - calibrated output. Return the diagnosis with an explicit confidence "
                        "band, and surface 'insufficient visual evidence' as a normal outcome.")

        # Keep the whole template on one page
        f.ensure(STYLE["mono"]["lead"] * (VERIFIER_PROMPT.count("\n") + 3) + 0.4)
        f.add("h2", "Verifier prompt template")
        f.add("mono", VERIFIER_PROMPT)

        # ── Caveats ───────────────────────────────────────────────────────────
        f._new_page()
        f.add("h1", "7. Caveats and next steps")
        f.add("h2", "What would weaken these conclusions")
        f.add("bullet", "Correctness is LLM-judged, not human-adjudicated. The judge is demonstrably "
                        "lenient toward vague answers on catch-all classes, which is why those rows are "
                        "reported separately; residual leniency inside the specific stratum is possible.")
        f.add("bullet", "Agreement between predictions is measured by content-word overlap, not by "
                        "resolving each prediction to a canonical class label. Two phrasings of the same "
                        "diagnosis may register as disagreement, which would inflate the instability rate.")
        f.add("bullet", "Intermediate-step evidence covers 10 datasets selected as the largest gainers and "
                        "regressors, not a random sample. Final-answer evidence covers all 26. The "
                        "presupposition finding should be confirmed on the full set.")
        f.add("bullet", "self_cot intermediate reasoning was never persisted, so all step-level findings "
                        "come from cot_yaml. Whether self_cot anchors as hard is untested.")

        f.add("h2", "Suggested order of work")
        f.add("bullet", "Re-run the disease track with existence-neutral step 2 wording on the healthy "
                        "subset only. This is a cheap, targeted experiment and it tests the single largest "
                        "effect in this report.")
        f.add("bullet", "Persist self_cot intermediate responses so the anchoring analysis can be repeated "
                        "for that condition.")
        f.add("bullet", "Implement Stage A and B, which need no new model, and measure end-to-end before "
                        "committing to a trained verifier.")
        f.add("bullet", "Only then consider a fine-tuned verifier, trained on the flagged-case "
                        "disagreements this pipeline produces.")

        f.add("h2", "Artifacts")
        f.add("note",
              "outputs/verifier_analysis/verifier_analysis.json - all statistics\n"
              "outputs/verifier_analysis/*.png - charts reproduced above\n"
              "src/verifier_analysis.py - analysis\n"
              "src/verifier_report.py - this report")

        f.close()


VERIFIER_PROMPT = """\
You are verifying an agricultural image diagnosis. You have the image, a proposed
diagnosis, and alternative diagnoses the same model produced when asked differently.

Proposed diagnosis: {claim}
Alternatives from independent probes: {alternatives}
Agreement score: {agreement}

Apply these rules in order. Do not skip to the answer.

1. HEALTHY CHECK - decide this before anything else.
   Look for actual abnormality: lesions, discoloration, wilting, insect damage.
   If the tissue looks normal, answer NO_ABNORMALITY. Do not let the diagnosis or
   the probe questions convince you symptoms exist. Describing a symptom is not
   evidence of a symptom.

2. EVIDENCE CHECK - is the claimed symptom visible?
   Name the specific pixels/regions supporting the claim. If you cannot point to
   a region, the claim is unsupported regardless of how confident it sounds.

3. DISAGREEMENT CHECK - the alternatives are the model's own answers.
   If they conflict, the visual evidence was not decisive. Prefer the diagnosis
   supported by point 2, or answer INSUFFICIENT_EVIDENCE.

4. CONFIDENCE-LANGUAGE WARNING.
   Ignore words like "clearly", "characteristic of", "definitively". In this
   model they are associated with WRONG answers, not right ones. Judge only the
   visible evidence.

5. IMAGE-QUALITY CHECK.
   If resolution, focus, or crop make diagnosis unsafe, answer
   INSUFFICIENT_EVIDENCE. This is a correct outcome, not a failure.

Return JSON:
{"verdict": "CONFIRM" | "REJECT" | "NO_ABNORMALITY" | "INSUFFICIENT_EVIDENCE",
 "corrected_diagnosis": "<diagnosis, or null>",
 "visual_evidence": "<region and what is visible, or null>",
 "confidence": "high" | "medium" | "low",
 "reasoning": "<2-3 sentences>"}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_dir", default="outputs/verifier_analysis/")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = Path(args.analysis_dir)
    res = json.loads((d / "verifier_analysis.json").read_text())
    out = Path(args.out) if args.out else d / "verifier_findings.pdf"
    build(res, d, out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
