"""
Decide whether each field ticket should be approved or rejected based on the
modelled likelihood of errors in its rows.

Logic:
- Train the Naive Bayes model from model_train.py on the provided dataset.
- Predict error probability for every row.
- A ticket is APPROVED only if all its rows are predicted error-free
  (probability < threshold); otherwise it is REJECTED.

Usage:
    python3 ticket_decider.py --threshold 0.9 --ticket-field title --top 10

Notes:
- Defaults: threshold=0.9, ticket_field="title".
- Requires pandas to read the CSV training file. Prefers loading a persisted model (naive_bayes_model.pkl) if present; otherwise trains on the fly.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List

from model_train import (
    DATA_PATH,
    MODEL_PATH,
    NUMERIC_FIELDS,
    NaiveBayes,
    build_dataset,
    load_model,
    load_table,
)


def decide(threshold: float, ticket_field: str, top: int) -> None:
    headers, body = load_table(DATA_PATH)
    if ticket_field not in headers:
        raise SystemExit(f"Ticket field '{ticket_field}' not found. Available: {headers}")

    data, labels = build_dataset(headers, body)
    print(
        f"Loaded {len(data)} rows; positives: {sum(labels)} "
        f"({sum(labels)/len(labels):.2%})"
    )

    # Prefer loading persisted model; fall back to training if missing.
    if MODEL_PATH.exists():
        try:
            model = load_model(MODEL_PATH)
        except Exception:
            model = NaiveBayes(NUMERIC_FIELDS)
            model.fit(data, labels)
    else:
        model = NaiveBayes(NUMERIC_FIELDS)
        model.fit(data, labels)

    # Predict probabilities
    probs = [model.predict_proba(row) for row in data]
    preds = [int(p >= threshold) for p in probs]

    # Aggregate by ticket
    decisions: Dict[str, Dict[str, object]] = defaultdict(lambda: {"rows": 0, "max_prob": 0.0, "any_error": False})
    for row, prob, pred in zip(data, probs, preds):
        ticket_id = (row.get(ticket_field) or "UNKNOWN").strip() or "UNKNOWN"
        rec = decisions[ticket_id]
        rec["rows"] += 1
        rec["max_prob"] = max(rec["max_prob"], prob)
        rec["any_error"] = rec["any_error"] or bool(pred)

    approved = {t: rec for t, rec in decisions.items() if not rec["any_error"]}
    rejected = {t: rec for t, rec in decisions.items() if rec["any_error"]}

    print(f"\nThreshold: {threshold}")
    print(f"Ticket field: {ticket_field}")
    print(f"Decisions: {len(approved)} approved, {len(rejected)} rejected (total {len(decisions)})")

    if top > 0 and rejected:
        print(f"\nTop {top} highest-risk tickets (by max row probability):")
        for ticket, rec in sorted(rejected.items(), key=lambda kv: kv[1]["max_prob"], reverse=True)[:top]:
            print(
                f"  ticket={ticket} rows={rec['rows']} max_prob={rec['max_prob']:.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Probability threshold for rejecting a ticket (tuned best is 0.9).",
    )
    parser.add_argument("--ticket-field", type=str, default="title", help="Column name representing the ticket ID.")
    parser.add_argument("--top", type=int, default=5, help="How many highest-risk tickets to display.")
    args = parser.parse_args()
    decide(args.threshold, args.ticket_field, args.top)


if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise SystemExit(f"Data file not found at {DATA_PATH}")
    main()
