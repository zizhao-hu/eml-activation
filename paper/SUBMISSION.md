# arXiv submission guide

I cannot submit to arXiv from this sandbox — submission requires your
arXiv account, an interactive web upload, and (for first-time `cs.LG`
authors) an endorsement. This file contains everything you need to do it
yourself in about 10 minutes.

## Files in this directory

| File | Purpose |
|---|---|
| `paper.tex` | Source LaTeX. |
| `refs.bib` | BibTeX database. |
| `paper.bbl` | Pre-built bibliography (arXiv requires this so it doesn't need to run BibTeX). |
| `neurips_2024.sty` | Lightweight NeurIPS-style stand-in. **Replace with the official `neurips_2024.sty` from `media.neurips.cc/Conferences/NeurIPS2024/Styles.zip` before final NeurIPS-conference submission.** For arXiv, the included one is fine. |
| `paper.pdf` | Compiled output (for your visual review only — don't upload). |
| `eml_attention_arxiv.tar.gz` | The single archive arXiv wants. |

## Before submitting

1. **De-anonymize.** Open `paper.tex`, find `\author{Anonymous...}`, replace with your real name, email, and affiliation. Recompile to verify (`pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper`), then rebuild the tarball:

   ```
   tar --exclude='*.aux' --exclude='*.log' --exclude='*.out' --exclude='*.blg' \
       -czf eml_attention_arxiv.tar.gz paper.tex refs.bib paper.bbl neurips_2024.sty
   ```

2. **Skim the PDF for accuracy.** Tables 1 and 2 are filled in from real runs but at small CPU scale (4 layers, d=128, 500 iters). The paper says so explicitly; if you have GPU results at larger scale, swap them in before submitting.

3. **Decide on co-authors.** The acknowledgements / contributions can stay as is.

## Submitting

1. Go to <https://arxiv.org/submit>. Log in (create an account if you don't have one).
2. Click **Start a new submission**.
3. Upload `eml_attention_arxiv.tar.gz`. arXiv will run its own LaTeX pipeline; the included `paper.bbl` means it doesn't need BibTeX. It should produce a 6-page PDF.
4. **Metadata** form fields:
   - **Title:** `Sheffer-Stroke Attention: A Single Binary Operator Replaces Softmax in Transformers`
   - **Authors:** your name(s) and affiliation(s).
   - **Abstract:** copy from `paper.tex` (the `\begin{abstract}...\end{abstract}` block).
   - **Primary category:** `cs.LG` (Machine Learning)
   - **Cross-list categories:** `cs.NE` (Neural and Evolutionary Computing), optionally `stat.ML`.
   - **Comments:** something like `6 pages, 2 tables. Code at https://github.com/zizhao-hu/llm-proof-walkthrough`.
   - **License:** I recommend `CC BY 4.0` for max reuse. Default `arXiv non-exclusive` is fine if you prefer.
5. **Endorsement.** If this is your first `cs.LG` submission, arXiv will say it needs endorsement. The fastest path: ask a colleague who has an arXiv `cs.LG` paper to endorse you (they go to arxiv.org/auth/endorse and enter the endorsement code arXiv shows you).
6. **Preview** the PDF that arXiv generated. Check page count, tables, references. If it's broken, fix `paper.tex` locally, rebuild the tarball, replace.
7. **Submit.** It usually goes live in the next daily mailing (US-Eastern late afternoon).

## Common issues

- **`! LaTeX Error: File 'natbib.sty' not found.`** — arXiv has natbib; you don't need to include it.
- **Bibliography missing.** — make sure `paper.bbl` is in the tarball; without it arXiv won't run BibTeX.
- **`neurips_2024.sty` claim disputed.** — the included file is a hand-rolled stand-in, not the official one. arXiv accepts it; NeurIPS itself does not. If you ever submit to NeurIPS proper, replace with the official style file.

## After it's live

- arXiv will assign an ID (e.g. `arXiv:2604.XXXXX`).
- Update the project README at `https://github.com/zizhao-hu/llm-proof-walkthrough` with the citation block.
- The `paper.tex` references arXiv 2603.21852 and 2604.13871 (Odrzywolek's two papers); double-check those IDs are still valid before going live.
