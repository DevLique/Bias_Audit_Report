# Bias Audit Report Notebook

This Jupyter notebook provides a workflow for auditing text datasets for bias using IBM AI Fairness 360 (AIF360) and standard NLP tools.

## Features

- **Single-text bias scan**: Flag identity terms, slurs, gendered language, and sentiment tilt in any text.
- **Dataset audit**: Analyze a CSV dataset with a `protected_attribute` column for representation and group-wise bias indicators.
- **Fairness metrics**: Train a baseline classifier and compute metrics such as Statistical Parity Difference, Disparate Impact, Equal Opportunity Difference, and Average Odds Difference.
- **Mitigation**: Apply reweighing to reduce bias and compare fairness metrics before and after mitigation.
- **Report generation**: Save concise, copy-pastable audit reports and charts to the `bias_audit_outputs` folder.

## Expected Dataset Format

CSV file with columns:
- `text`: The content to analyze (string)
- `protected_attribute`: Group membership (string or int, e.g., `female`/`male`)
- `label`: (Optional) Ground-truth outcome for fairness metrics (0/1 or convertible)

## Quick Start

1. **Install dependencies** (uncomment in notebook if needed):
    ```python
    # !pip install --quiet aif360==0.6.1 scikit-learn pandas numpy matplotlib nltk imbalanced-learn
    ```

2. **Run notebook cells** in order:
    - Paste or load your dataset (`my_dataset.csv`).
    - Inspect representation and run bias scans.
    - Train and evaluate the classifier if `label` exists.
    - Apply mitigation and compare metrics.
    - Generate and save the audit report.

## Outputs

- **Charts**: Fairness metrics visualizations.
- **Text files**: Model performance and audit summary.
- **Folder**: All outputs saved to `bias_audit_outputs`.

## Notes

- For production use, expand the slur/offensive lexicon and review identity terms.
- Metrics are computed using AIF360 and scikit-learn.
- Review single-text flags for context and language cues not captured by model-level metrics.

## License

See repository for license details.
