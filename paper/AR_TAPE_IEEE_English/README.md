# AR-TAPE: English LaTeX manuscript

This is a conference-style, two-column research manuscript using the standard `IEEEtran` conference class. It is prepared for academic evaluation, not represented as an accepted IEEE publication.

## Build

Open `main.tex` as the main file in Overleaf, or run locally:

```sh
latexmk -pdf main.tex
```

Alternatively, use `pdflatex main`, `bibtex main`, and two further `pdflatex main` passes. Tectonic can compile the project with `tectonic main.tex` and handles the bibliography automatically.

The project uses standard TeX Live packages: IEEEtran, amsmath, amssymb, graphicx, booktabs, cite, TikZ, PGFPlots, and hyperref. No shell escape or proprietary fonts are required.

## Files

- `sections/`: complete English manuscript.
- `figures/`: editable TikZ architecture/training diagrams, PGFPlots chart, and paired reconstruction layout.
- `tables/`: booktabs tables generated from audited measurements.
- `images/`: cropped panels from saved experimental grids, without generated reconstructions or enhancement.
- `data/results_snapshot.json`: numerical source snapshot, seed-level records, source paths, and SHA-256 hashes.
- `data/sweep.dat`, `data/seeds.dat`: plot-ready aggregates and individual-seed values.
- `data/figure_provenance.json`: reconstruction source mapping and pairing verification.
- `references.bib`: bibliography.

The project snapshot is `be737ce33b219f22b7ff86f9b17d04c7e6ebdd4b`. Data were audited on October 25, 2026. No model training was run to create the manuscript.

The paper explicitly documents test-time BatchNorm contamination, the directory/metadata mismatch in one FSHA group, missing medical experiments, and the unmet original utility target. These are limitations of the current evidence and are not corrected retrospectively in the raw outputs.
