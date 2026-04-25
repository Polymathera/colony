---
name: scientific-debugging
description: |
  Structured root-cause analysis worksheet that walks the agent
  through the scientific method for debugging: observe → hypothesise
  → predict → experiment → revise. Use this when the user reports a
  bug or unexpected behaviour and you want to avoid jumping to a
  conclusion before evidence supports it.
when_to_use: |
  Triggered when the user describes a defect ("it's broken", "this
  used to work", "intermittent failure") and wants a disciplined
  diagnosis path. Also useful when several plausible causes compete
  and the agent needs to design experiments that discriminate
  between them.
sandbox_image_role: default
script: scripts/run.sh
params:
  bug_description:
    type: string
    required: true
    description: |
      One-paragraph description of the observed bug or unexpected
      behaviour, in the user's own words.
  context:
    type: string
    required: false
    description: |
      Optional extra context: stack trace excerpt, recent commits
      suspected to be related, environment hints, prior diagnoses
      already ruled out.
  hypotheses:
    type: string
    required: false
    description: |
      Optional comma-separated list of initial hypotheses the agent
      already has in mind. The worksheet uses them as the seed list
      for the falsification phase.
timeout_seconds: 60
---

# Scientific Debugging

Emits a Markdown worksheet that imposes a five-stage discipline on
diagnosis:

1. **Observation** — what is actually happening, free of inference.
2. **Hypotheses** — every plausible cause, *with* explicit
   "what would falsify this" predictions.
3. **Experiments** — concrete, ordered checks the agent can run
   (logs, breakpoints, bisection, isolated repro). Each experiment
   must discriminate between hypotheses.
4. **Evidence** — facts collected, mapped back to which hypothesis
   they support or refute.
5. **Conclusion** — surviving hypothesis or revised model. Includes
   a "what we still don't know" section so the agent stays honest.

The worksheet is meant to be read by the LLM and used as a planning
scaffold, not as a static document. The agent should fill in the
sections as it gathers evidence and revisit them when new findings
arrive.
