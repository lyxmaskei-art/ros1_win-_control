# RAS Overleaf package

This folder is a self-contained LaTeX project prepared for `Robotics and Autonomous Systems`.

Contents:

- `main.tex`: main manuscript entry.
- `sections/`: section files.
- `references.bib`: bibliography.
- `figures/`: copied result figures used by the manuscript.
- `highlights.txt`: Elsevier-style highlights draft.
- `experiment_summary.json`: structured offline metrics exported from the integrated scripts.

Recommended local compile sequence:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Notes:

- The manuscript currently reports offline numerical experiments and live CoppeliaSim runs.
- Physical robot experiments are intentionally described as future work only.
- Author names and affiliation are prefilled from the local DLCCZNN manuscript and can be edited if needed before submission.
