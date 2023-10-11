# BiasX

**Warning: contains data/content that may be upsetting or offensive**

## Setting up

1. Install python dependencies: `pip install -e .`
2. Download SBIC data: `bash scripts/download-data.sh`

## Fine-tuning a DeBERTa toxicity classifier on SBIC

`python src/classify/classification.py configs/classify/deberta.yaml` fine-tunes a deberta-v3-large model on SBIC training set, and produces predictions for the test set.

## Generating explanations with GPT-3.5

`python src/generation/in-context-generation.py configs/generate/text-davinci-003.yaml`
produces a file with social-bias-frames-style explanations:
```json
...
{
  "post": "No, can you get one of the boys to carry that out? It's too heavy for you.",
  "offensiveYN": true,
  "referenceMinorityGroups": [
    "women"
  ],
  "referenceStereotypes": [
    "women aren't strong.",
    "women can't lift things."
  ],
  "offensivePrediction": 1,
  "generatedMinorityGroup": "women",
  "generatedStereotype": "Implies that women are weaker than men and not capable of carrying out certain tasks."
}
```

## License

MIT

## Citation

Coming soon.