import argparse
import numpy as np
import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '.')
from src.active_learning import ActiveLearner, ActiveLearningSession

logging.basicConfig(level=logging.WARNING)   


R  = "\033[91m"   # red
G  = "\033[92m"   # green
Y  = "\033[93m"   # yellow
B  = "\033[94m"   # blue
M  = "\033[95m"   # magenta
C  = "\033[96m"   # cyan
W  = "\033[97m"   # white
DIM= "\033[2m"
RST= "\033[0m"

BRAIN_REGIONS = [
    "frontal cortex",     "motor cortex",        "somatosensory cortex",
    "hippocampus",        "entorhinal cortex",   "amygdala",
    "thalamus",           "hypothalamus",        "midbrain",
    "cerebellum",         "brainstem",           "olfactory bulb",
    "visual cortex",      "auditory cortex",     "parietal cortex",
    "edge/artifact",      "unknown",
]


def print_header() -> None:
    print(f"\n{B}{'='*60}{RST}")
    print(f"{W}  Brain Image Active Learning — Interactive Labeling Tool{RST}")
    print(f"{B}{'='*60}{RST}\n")


def print_query(q: dict, cluster_labels_used: list[int]) -> None:
    is_disagreement = q['current_label'] != q['predicted_label']
    conf_color = R if q['confidence'] < 50 else Y if q['confidence'] < 75 else G

    print(f"\n  {M}Query {q['query_number']}{RST} — {DIM}{q['filename']}{RST}")
    print(f"  {'─'*50}")
    print(f"  Current cluster label : {Y}{q['current_label']}{RST}")
    print(f"  Model predicts        : {conf_color}{q['predicted_label']}{RST} "
          f"({conf_color}{q['confidence']}% confident{RST})")
    if is_disagreement:
        print(f"  {R}⚠  Model disagrees with current label!{RST}")
    print(f"  Uncertainty (entropy) : {q['entropy']:.4f}  "
          f"{DIM}(higher = more uncertain){RST}")
    print(f"\n  {DIM}Top 3 predicted classes:{RST}")
    for item in q['top_3_classes']:
        bar = '█' * int(item['prob'] / 5)
        print(f"    Cluster {item['class']:2d}  {bar:<20s}  {item['prob']:.1f}%")
    print(f"\n  {DIM}Image: {q['image_path']}{RST}")


def get_label_from_researcher(q: dict, n_classes: int) -> dict:
    print(f"\n  {C}What is the correct cluster for this slice?{RST}")
    print(f"  {DIM}Press Enter to CONFIRM current label ({q['current_label']}), "
          f"or type a number 0-{n_classes-1} to CORRECT it.{RST}")
    print(f"  {DIM}Type 's' to skip this slice.{RST}")

    while True:
        try:
            raw = input(f"  {G}Your answer [{q['current_label']}]: {RST}").strip()

            if raw == 's':
                return None   # skip

            if raw == '':
                confirmed_label = q['current_label']
            else:
                confirmed_label = int(raw)
                if not (0 <= confirmed_label < n_classes):
                    print(f"  {R}Invalid. Enter a number between 0 and {n_classes-1}.{RST}")
                    continue

            
            print(f"\n  {DIM}Optional: name the brain region "
                  f"(e.g. hippocampus, cerebellum). Press Enter to skip.{RST}")
            for i, r in enumerate(BRAIN_REGIONS, 1):
                print(f"    {i:2d}. {r}")
            raw_region = input(f"  {G}Region number or name [skip]: {RST}").strip()

            region = ""
            if raw_region.isdigit():
                idx = int(raw_region) - 1
                if 0 <= idx < len(BRAIN_REGIONS):
                    region = BRAIN_REGIONS[idx]
            elif raw_region:
                region = raw_region

            return {
                'label'       : confirmed_label,
                'brain_region': region,
                'entropy'     : q['entropy'],
            }

        except ValueError:
            print(f"  {R}Please enter a valid number.{RST}")
        except KeyboardInterrupt:
            print(f"\n  {Y}Interrupted. Saving progress...{RST}")
            return None


def print_round_summary(result: dict, corrections: dict) -> None:
    n_corrected = sum(
        1 for fname, c in corrections.items()
        if c['label'] != next(
            (q['current_label'] for q in result['queries']
             if q['filename'] == fname), c['label']
        )
    )
    n_confirmed = len(corrections) - n_corrected

    print(f"\n  {B}{'─'*50}{RST}")
    print(f"  {W}Round {result['round']} summary:{RST}")
    print(f"    Confirmed correct : {G}{n_confirmed}{RST}")
    print(f"    Corrected         : {R}{n_corrected}{RST}")
    print(f"    Classifier accuracy (on labeled set): "
          f"{result['accuracy']:.1%}")
    print(f"    Total labeled so far: {result['n_labeled']}")
    print(f"  {B}{'─'*50}{RST}")


def print_final_report(report: dict) -> None:
    print(f"\n{B}{'='*60}{RST}")
    print(f"{W}  Final Session Report{RST}")
    print(f"{B}{'='*60}{RST}")
    print(f"  Session ID       : {report['session_id']}")
    print(f"  Rounds completed : {report['rounds_completed']}")
    print(f"  Total queries    : {report['total_queries']}")
    print(f"  Labels collected : {report['total_labeled']}")
    print(f"  Corrections made : {report['total_corrections']} "
          f"({report['correction_rate']:.1%} of labeled)")
    print(f"  Final accuracy   : "
          f"{report.get('final_accuracy', 'N/A')}")
    print(f"  Converged        : {G+'Yes' if report['converged'] else R+'No'}{RST}")

    if report['region_annotations']:
        print(f"\n  {W}Brain region annotations:{RST}")
        for region, count in sorted(report['region_annotations'].items(),
                                    key=lambda x: -x[1]):
            bar = '█' * count
            print(f"    {region:<25s} {bar}  ({count})")

    if len(report['accuracy_history']) > 1:
        print(f"\n  {W}Accuracy per round:{RST}")
        for i, acc in enumerate(report['accuracy_history'], 1):
            bar = '█' * int(acc * 30)
            print(f"    Round {i}: {bar:<30s} {acc:.3f}")
    print(f"\n{B}{'='*60}{RST}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive active learning for brain image clustering."
    )
    parser.add_argument('--n_query',  type=int, default=10,
                        help="Slices to show per round (default: 10)")
    parser.add_argument('--rounds',   type=int, default=999,
                        help="Max rounds to run (default: until convergence)")
    parser.add_argument('--resume',   type=str, default="",
                        help="Resume a saved session by ID")
    parser.add_argument('--features', type=str,
                        default="results/embeddings/features.npy")
    parser.add_argument('--labels',   type=str,
                        default="results/embeddings/cluster_labels.npy")
    parser.add_argument('--filenames',type=str,
                        default="results/embeddings/filenames.npy")
    args = parser.parse_args()

    SESSION_DIR = Path("results/active_learning_sessions")
    SESSION_DIR.mkdir(parents=True, exist_ok=True)

    print_header()

    for p in [args.features, args.labels, args.filenames]:
        if not os.path.exists(p):
            print(f"{R}Error: file not found: {p}{RST}")
            print(f"{DIM}Run the main pipeline first: python run_pipeline.py{RST}")
            sys.exit(1)

    learner = ActiveLearner.from_embeddings(
        features_path  = args.features,
        labels_path    = args.labels,
        filenames_path = args.filenames,
    )

    if args.resume:
        session_path = SESSION_DIR / f"{args.resume}.json"
        if not session_path.exists():
            print(f"{R}Session file not found: {session_path}{RST}")
            sys.exit(1)
        session = ActiveLearningSession.Load(str(session_path))
        print(f"{G}Resumed session '{args.resume}' "
              f"(Round {session.rounds_completed}, "
              f"{len(session.labeled_samples)} labels){RST}\n")
    else:
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"session_{ts}"
        session    = learner.start_session(session_id)
        print(f"{G}New session: {session_id}{RST}")
        print(f"{DIM}{len(learner.features)} slices | "
              f"{learner.n_classes} clusters{RST}\n")

    for round_num in range(args.rounds):
        if learner.convergence_check(session):
            print(f"\n{G}Model has converged after {session.rounds_completed} rounds.{RST}")
            break

        print(f"\n{B}--- Round {session.rounds_completed + 1} ---{RST}")
        print(f"{DIM}Finding the {args.n_query} most uncertain slices...{RST}")

        result = learner.run_round(session, n_query=args.n_query)

        if not result['queries']:
            print(f"{G}No more uncertain slices. Labeling complete.{RST}")
            break

        print(f"\n{W}Classifier accuracy on labeled set: "
              f"{result['accuracy']:.1%}{RST}")
        print(f"{DIM}Labeled: {result['n_labeled']} | "
              f"Unlabeled: {result['n_unlabeled']}{RST}")

        corrections = {}

        for q in result['queries']:
            print_query(q, list(range(learner.n_classes)))

            correction = get_label_from_researcher(q, learner.n_classes)
            if correction is not None:
                corrections[q['filename']] = correction
            else:
                print(f"  {DIM}Skipped.{RST}")

        if not corrections:
            print(f"\n{Y}No labels submitted this round. Exiting.{RST}")
            break

        session = learner.apply_corrections(session, corrections)
        print_round_summary(result, corrections)

        session_path = SESSION_DIR / f"{session.session_id}.json"
        session.save(str(session_path))
        print(f"  {DIM}Session saved → {session_path}{RST}")

        try:
            cont = input(f"\n  {C}Continue to next round? [Y/n]: {RST}").strip().lower()
            if cont in ('n', 'no', 'q', 'quit'):
                break
        except KeyboardInterrupt:
            break

    if len(session.labeled_samples) > 0:
        print(f"\n{B}Applying corrections to clustering...{RST}")
        new_labels = learner.retrain_clustering(session)
        out = "results/embeddings/cluster_labels_refined.npy"
        np.save(out, new_labels)
        print(f"{G}Refined labels saved → {out}{RST}")

        audit = [
            {'filename': s.filename, 'original': s.original_label,
             'corrected': s.human_label, 'region': s.brain_region,
             'round': s.round_number}
            for s in session.labeled_samples
        ]
        audit_path = SESSION_DIR / f"{session.session_id}_audit.json"
        with open(audit_path, 'w') as f:
            json.dump(audit, f, indent=2)
        print(f"{G}Audit trail saved → {audit_path}{RST}")

    report = learner.generate_report(session)
    print_final_report(report)

    report_path = SESSION_DIR / f"{session.session_id}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"{DIM}Full report → {report_path}{RST}\n")


if __name__ == "__main__":
    main()
