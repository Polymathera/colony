"""Emit a scientific-method root-cause-analysis worksheet.

The worksheet is read by the LLM, not by a human — its purpose is to
impose a five-stage discipline on diagnosis: observe, hypothesise,
predict, experiment, conclude. Each section starts seeded with the
caller-provided context so the agent can fill the rest in subsequent
turns.

Standard library only.
"""

from __future__ import annotations

import argparse
import sys
import textwrap


_TEMPLATE = """\
# Root-Cause Analysis Worksheet

## 0. Bug under investigation

{bug_description}

{context_section}

## 1. Observation

> *Stage rule: write what you have actually seen. No inference, no
> assumed cause yet. Quote logs verbatim, paste exact error
> messages.*

- (write the literal symptoms here — "page returns 500", "test
  passes locally, fails in CI", etc.)

## 2. Hypotheses

> *Stage rule: enumerate every plausible cause, even ones you doubt.
> For each, write the prediction it makes that no rival hypothesis
> shares — the **falsifier**. A hypothesis without a falsifier is a
> story, not a hypothesis.*

| # | Hypothesis | Falsifier (what evidence would rule it out?) |
| - | ---------- | -------------------------------------------- |
{hypothesis_rows}

## 3. Experiments

> *Stage rule: design experiments that **discriminate** between the
> hypotheses above. Order them by (information yield) ÷ (cost). The
> first experiment should rule out at least one hypothesis no matter
> how it turns out.*

1. (cheapest discriminating check — e.g., "git bisect between known-
   good and known-bad commits")
2. (next-cheapest — e.g., "isolate the failing input and run with
   verbose logging")
3. ...

## 4. Evidence

> *Stage rule: record what the experiments produced and which
> hypothesis each piece of evidence supports or refutes. If an
> experiment failed to discriminate, replace it with a sharper one.*

| Experiment | Result | Supports | Refutes |
| ---------- | ------ | -------- | ------- |
| | | | |

## 5. Conclusion

> *Stage rule: name the surviving hypothesis. Then list everything
> you still don't know — those are the next iteration's experiments,
> not pretend-certainty.*

- **Most likely cause:**
- **Confidence:**
- **Open questions:**
- **Fix proposal:**
- **How to verify the fix discriminates from a coincidental
  improvement:**
"""


def _format_hypotheses(raw: str | None) -> str:
    """Build the table rows from a comma-separated seed list.

    Empty seed list → one empty row so the LLM sees the table shape.
    """
    if not raw or not raw.strip():
        return "| 1 | (write your first hypothesis here) | (what would rule it out?) |"
    rows: list[str] = []
    for i, h in enumerate(
        [s.strip() for s in raw.split(",") if s.strip()],
        start=1,
    ):
        rows.append(f"| {i} | {h} | (what would rule this out?) |")
    return "\n".join(rows)


def _format_context(context: str | None) -> str:
    if not context or not context.strip():
        return ""
    return (
        "## Context provided by the user\n\n"
        + textwrap.indent(context.strip(), "> ")
        + "\n"
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a scientific-method RCA worksheet.",
    )
    parser.add_argument("--bug_description", required=True)
    parser.add_argument("--context", default="")
    parser.add_argument("--hypotheses", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    print(_TEMPLATE.format(
        bug_description=args.bug_description.strip(),
        context_section=_format_context(args.context),
        hypothesis_rows=_format_hypotheses(args.hypotheses),
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
