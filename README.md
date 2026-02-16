# ENSO-WindPerturb-Thermocline

## Modulation of Thermocline Feedback by Westerly Wind Bursts and Surface Wind Forcing During El Niño Development in an Ocean-Only Model

This repository contains all processing and plotting code used in the associated research article.

## Environment

```bash
dconda env create -f environment.yml
conda activate enso_windperturb_thermocline
```

## Directory Structure

```
data/
figures/
data_preprocessing.ipynb
utils.py
plotting.ipynb
environment.yml
```

Large files (>1GB, NetCDF and otherwise) in `./data` are not included in the repository. Contact the author if needed.

## Data Sources

* CMIP6 HighResMIP simulations (CEDA Archive)
  [https://hrcm.ceda.ac.uk/research/cmip6-highresmip/](https://hrcm.ceda.ac.uk/research/cmip6-highresmip/)
* NPD eORCA025 simulations (NOC/MSM)
* JRA55-do (Copernicus Climate Data Store)
* ORAS5 ocean reanalysis
* EN4 observational dataset
* NOAA Climate Prediction Centre Oceanic Niño Index v2 

## Workflow

Run the following in order (requires data access):

1. TBD
2. TBD
3. TBD

## Contact

**Sreevathsa Golla** – [sg13n23@soton.ac.uk](mailto:sg13n23@soton.ac.uk)

### Acknowledgements

Portions of the repository (code formatting/tidying up and documentation cleanup) were assisted using generative AI tools (i.e., ChatGPT 5.2 and Claude Sonnet 4.5). All scientific content, analysis logic, and results were developed by the primary author. Please feel free to contact the author if you require any further clarification.

## License

This repository is licensed under the MIT License. See the LICENSE file for details.
