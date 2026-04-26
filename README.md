# Proof State Walkthrough

**Theme:** Education / Learning Tool for Students and Outsiders
**One-liner:** A web app that segments any mathematical proof into steps, tracks what's known vs. still-to-show at each line, and lets you click any step to interrogate it in plain English.

## Problem

Textbook proofs are written for people who already understand them. Each line is terse, assumes background, and never explains *why* that specific move was the right one. Students spend 45 minutes on a single line of a real analysis proof trying to figure out where it came from. No existing tool goes beyond "here's a re-explanation of the same proof" — none of them track the evolving logical state.

## The sketch

User pastes a proof in LaTeX or plain text. A FastAPI backend sends it to Claude with a structured-output prompt that (1) segments the proof into numbered steps, (2) labels the technique at each step (e.g., "triangle inequality", "induction hypothesis application", "substitution"), and (3) produces a proof-state record per step: a two-column snapshot of what has been established vs. what remains to be shown. The React frontend renders a side-by-side panel: raw proof on the left, annotated walkthrough on the right, with the proof-state diff highlighted between steps. Clicking any step opens a seeded chat window with "Why this step?" pre-filled; the chat context includes the full proof, the current step's annotation, and the proof state before and after it.

## Why now

Claude and GPT-4o are reliable enough on undergraduate-level math to get technique labels right the vast majority of the time. Structured JSON output makes the segmentation pipeline deterministic enough to be useful. MathJax v3 handles arbitrary LaTeX in the browser without friction. None of that was true two years ago.

## Demo surface

Paste a 10-step proof of Cauchy-Schwarz. The right panel shows each step labeled with its technique, the proof-state columns updating as you scroll. Click step 4, ask "why not just apply AM-GM directly here?", and get a one-paragraph answer explaining the constraint that forces this detour.

## Risks / honest take

LLMs hallucinate confidently in math above undergraduate level — technique labels will be wrong on graduate proofs in ways that actively mislead students. The honest scope is calculus, linear algebra, and intro real analysis; beyond that you'd need a Lean/Coq backend to verify the proof structure before annotating it.

## Stack guess

Python/FastAPI, React + MathJax v3, Claude claude-sonnet-4-6 with structured JSON output for step segmentation, streamed chat for interrogation

---

_Spawned from [auto-brainstorm](https://github.com/zizhao-hu/auto-brainstorm) on 2026-04-26. Theme: `education`. This repo is auto-generated — if no one stars, forks, or files an issue within 7 days, it gets garbage-collected._
