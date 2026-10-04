# basalt-co2-4d-sensitivity

Code and results for the manuscript:

> J. A. Teruya, "Rock-Physics and Time-Lapse Seismic Sensitivity to CO₂ Mineralization
> in Basalt: Cementation, Non-Uniqueness and Detectability for the Paraná Basin, Brazil,"
> submitted, 2026.

The study models how CO₂ injection and mineral precipitation change the elastic
properties of a fractured basalt reservoir, and whether the change can be seen with
time-lapse (4D) seismic. All figures and numbers in the manuscript are produced by the
scripts in this repository.

## Contents

| Path | Description |
|---|---|
| `code/rp.py` | Rock physics: Gassmann, Reuss/Wood and patchy fluid mixing, phenomenological cementation, Hashin–Shtrikman bound, inverse Gassmann. All parameters. |
| `code/analysis0d.py` | Single-cell analysis: fluid substitution, frame stiffness, f∞–τ sweep, Monte Carlo (N = 20 000), non-uniqueness (S–f), Vs discrimination. |
| `code/model2d.py` | 2-D reservoir model: elastic change maps, convolutional 4D seismic, time shifts, noise test, filter-band selection, synthetic wells. |
| `code/figures.py` | Figures 1–10 of the manuscript (PDF and 600-dpi PNG). |
| `charla_ccs/modelos.py` | Reservoir geometry (`geology('basalto')`) and convolutional modelling (`synth`). |
| `results/` | Numerical outputs (`*.json`, `*.npz`). |
| `figures/` | Figures as published in the manuscript. |

## Reproduce

Python 3.11 was used. The full run takes under a minute on a laptop.

```bash
git clone https://github.com/USUARIO/basalt-co2-4d-sensitivity.git
cd basalt-co2-4d-sensitivity
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd code
python model2d.py      # writes results/model2d.*
python analysis0d.py   # writes results/analysis0d.*
python figures.py      # writes figures/fig01–fig10
```

Random seeds are fixed (Monte Carlo 20261001; time bands 7; noise test seeds
910–914; filter-selection seeds 110–112), so the outputs are reproducible.

## Main parameters

- Grain: K = 80.1 GPa, μ = 31 GPa, ρ = 2800 kg/m³.
- Brine: 2.237 GPa / 1040 kg/m³; supercritical CO₂: 0.159 GPa / 832 kg/m³.
- Massive basalt and vesicular facies calibrated to SAG-P2/P3 well medians; the
  fractured corridor (φ = 4.5 %) is a modelling assumption.
- Cemented end member = calibrated massive basalt (inside the Hashin–Shtrikman bound).

## Citation

Please cite the article and this software (see `CITATION.cff`, or the
"Cite this repository" button on GitHub). DOI: see the Zenodo badge once released.

## License

Code: MIT. Results and figures: CC BY 4.0. See `LICENSE`.

## Contact

Jorge A. Teruya — jteruya@usp.br — ORCID [0000-0003-3537-040X](https://orcid.org/0000-0003-3537-040X)
