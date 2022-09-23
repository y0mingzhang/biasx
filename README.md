# SBF-Modeling

## Prerequisites

- `bash scripts/download-data.sh` to download SBIC.v2 dataset.
- Use `conda env create -f environment.yml` to setup the conda environment.

## Run modeling (T5-Large)

Run `python src/main.py configs/t5-large.yaml`.

## Results (T5-Large)

```json
// dev eval:
{
  "offensive-accuracy": 0.9007356454,
  "offensive-F1": 0.9368120173,
  "minority-group-exact-match": 0.7965988344,
  "minority-group-BLEU-1": 0.8211781979,
  "minority-group-ROUGE-L": 0.8232424855,
  "stereotype-exact-match": 0.2937804529,
  "stereotype-BLEU-1": 0.5956009626,
  "stereotype-ROUGE-L": 0.6071415544
}

// test eval:
{
  "offensive-accuracy": 0.9055910403,
  "offensive-F1": 0.9417166316,
  "minority-group-exact-match": 0.8152944265,
  "minority-group-BLEU-1": 0.8423092961,
  "minority-group-ROUGE-L": 0.8388224244,
  "stereotype-exact-match": 0.2722897891,
  "stereotype-BLEU-1": 0.6057417989,
  "stereotype-ROUGE-L": 0.6062927246
}
```

[Qualitative Results](https://docs.google.com/spreadsheets/d/1dKMdfW3hAVTSRJa5twRhWReyf8vXzpDIbxfAXrFCUYs/edit#gid=1679411607)
