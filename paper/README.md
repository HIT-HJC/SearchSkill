# Paper Source

This directory contains the LaTeX source used for the SearchSkill paper.

Compile from this directory:

```bash
latexmk -pdf main.tex
```

If `latexmk` is unavailable, run:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The repository includes only the source files and figures needed to compile the paper, not generated auxiliary files.
