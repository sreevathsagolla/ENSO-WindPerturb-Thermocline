# ENSO-WindPerturb-Thermocline

## Wind-forced Modulation of Thermocline Feedback Processes During El Niño Development

This repository contains all processing and plotting code used in the associated research article.

## Environment

```bash
conda env create -f environment.yml
conda activate enso_windperturb_thermocline
```

## Directory Structure

```
data/
    EXP_SETUP_FILES/    # NEMO XIOS XMLs for the baseline and experiment runs*
    WND_FLDS/           # Perturbed uas/vas forcing, wind baselines and correction factors
    TOCE_CON/           # Conservative temperature, Tropical Pacific (1m and 1d)
    NINO_BOXES/         # Conservative temperature cropped to the Niño 1+2, 3, 3.4 and 4 boxes
    Z20/                # Depth of the 20°C isotherm (Z20)
    VOL_HEAT_BUDGETS/   # Daily volume and heat budgets from nemo_box_budget.py
    *.csv               # NOAA ONI v2 and Niño 3.4 SST anomalies (1976-2023)
    mesh_mask.nc / domain_cfg_HG3_eORCA12.nc  # eORCA025 and eORCA12 grid metrics
figures/                # Figures generated from plotting.ipynb
wind_perturbations.ipynb
preprocessing.ipynb
plotting.ipynb
nemo_box_budget.py
utils.py
environment.yml
```

Except for `data/VOL_HEAT_BUDGETS` and the two `.csv` files, files in `./data` are too large to host on GitHub (check .gitignore). Contact the author if needed.

## Data Sources

* HadGEM3-GC31-HH control-1950 (Roberts et al., 2019) simulation of CMIP6 HighResMIP (CEDA Archive)
  [https://hrcm.ceda.ac.uk/research/cmip6-highresmip/](https://hrcm.ceda.ac.uk/research/cmip6-highresmip/)
* NPD eORCA025 simulations (Blaker et al., 2025; National Oceanography Centre, UK)
* JRA55-do (Copernicus Climate Data Store)
* ORAS5 ocean reanalysis
* EN4 observational dataset
* NOAA PSL Oceanic Niño Index v2

## Experiments

Run labels used throughout `./data` and the notebooks:

* `ANWSUP` - ANomalous Westerlies (ANW) suppressed in the 2023 El Niño year
* `ANWTRN-JRA` / `ANWTRN-HG3` - Anomalous westerlies from 2023 JRA55-do / HG3-1995 transplanted onto neutral-year (2013) winds
* `WNDREP-HG3` / `WNDREP-HG3c` - 2013 winds globally replaced with HG3-1995 winds, without / with mean-state bias correction
* `CTRL_2013`, `CTRL_2023` - Unperturbed JRA55-do control equivalents
* `BASELINE_NPD` (1996-2023), `BASELINE_HG3` (1976-2005) - Climatological baselines; `HG3-1995` is the single El Niño year drawn from the latter

## Model version

This work is based on the NOC Near Present Day configuration (Blaker et al., 2025), which uses NEMO v4.2 (Madec et al., 2022).

- Repository: https://github.com/NOC-MSM/NOC_Near_Present_Day
- Branch: `main`
- Commit: `c9dc451122c73f2ac6683839cc0ca6977936769d` (2024-05-08)
- Retrieved: 2024-06-25

To obtain this version:

    git clone https://github.com/NOC-MSM/NOC_Near_Present_Day.git
    cd NOC_Near_Present_Day
    git checkout c9dc451122c73f2ac6683839cc0ca6977936769d

Further modifications to this version, done for the work presented in this repository, are described in the XML files in `data/EXP_SETUP_FILES`.

## Workflow

Run the following in order (requires data access); the NEMO experiments themselves are run between steps 1 and 2, configured using the XMLs in `data/EXP_SETUP_FILES`.

1. `wind_perturbations.ipynb` - builds the wind baselines, correction factors and the perturbed `uas`/`vas` forcing files for the five experiments.
2. `preprocessing.ipynb` - baselines, Niño-box and Tropical Pacific subsetting, Z20, and daily volume/heat budgets (via `nemo_box_budget.py`).
3. `plotting.ipynb` - generates all manuscript and Supporting Information figures (helpers in `utils.py`).

### References

- Blaker, A., Tooth, O., Palmiéri, J., Coward, A., & Mecking, J. (2025). NOC-MSM/NOC_Near_Present_Day: v0.9.0. Zenodo. https://doi.org/10.5281/zenodo.15310354
- Madec, G., Bourdallé-Badie, R., Chanut, J., Clementi, E., Coward, A., Ethé, C., Iovino, D., Lea, D., Lévy, C., Lovato, T., Martin, N., Masson, S., Mocavero, S., Rousset, C., Storkey, D., Müeller, S., Nurser, G., Bell, M., Samson, G., Mathiot, P., Mele, F., & Moulin, A. (2022). NEMO ocean engine, v4.2. Zenodo. https://doi.org/10.5281/zenodo.6334656
- Roberts, M. J., Baker, A., Blockley, E. W., Calvert, D., Coward, A., Hewitt, H. T., Jackson, L. C., Kuhlbrodt, T., Mathiot, P., Roberts, C. D., Schiemann, R., Seddon, J., Vannière, B., and Vidale, P. L.: Description of the resolution hierarchy of the global coupled HadGEM3-GC3.1 model as used in CMIP6 HighResMIP experiments, Geosci. Model Dev., 12, 4999–5028, https://doi.org/10.5194/gmd-12-4999-2019, 2019.

### Acknowledgements

Portions of this repository underwent code refactoring and documentation cleanup assisted by generative AI tools (Claude Opus 4.8 and Opus 5). All scientific content, analysis logic, and results were developed by the primary author. The plotting and core analysis scripts were verified after refactoring by confirming that outputs matched those produced beforehand. The preprocessing scripts, which produce large datasets and could not feasibly be re-run in full, were manually reviewed to confirm that only formatting and documentation were changed, with no modification to the code logic. Please feel free to contact the author if you require any further clarification or would like to raise any issues.

## Contact

**Sreevathsa Golla** - [sg13n23@soton.ac.uk](mailto:sg13n23@soton.ac.uk)
Website: [www.sreevathsagolla.com](www.sreevathsagolla.com)

## License

This repository is licensed under the MIT License. See the LICENSE file for details.
