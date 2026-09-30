# DA331 Assignment 5: CASAS Smart-Home Activity Dataset
**Kumar Utkarsh, 240150016**

| Task | Deliverable |
|---|---|
| 1. EDA notebook | [`01_EDA_CASAS.ipynb`](01_EDA_CASAS.ipynb) (executed, with outputs) |
| 2. Next-activity temporal model | [`02_Next_Activity_Model.ipynb`](02_Next_Activity_Model.ipynb) (executed, with outputs) |
| 3. Report (2 pages) | [`Report_240150016_Kumar_Utkarsh.pdf`](Report_240150016_Kumar_Utkarsh.pdf) (source: `report/report.html`) |

`figures/` holds the plots saved by the notebooks and used in the report.

## Data
Kaggle mirror of CASAS hh101 ([Zenodo 15708568](https://zenodo.org/records/15708568)):

```
CASAS-hh101/data/hh101.csv      raw sensor stream (1.29M events, 2012-07-18 to 2013-07-25)
CASAS-hh101/labeled/hh101.csv   annotated events (222,858 events, 2012-07-20 to 2012-09-17)
```

## Running
On Kaggle, attach the dataset and choose **Run All**; the notebooks find the files under `/kaggle/input`.
To run locally, put `CASAS-hh101/` inside a `data/` folder next to the notebooks.
Requirements: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib.

## Results (test period: last 8 days, 402 transitions, 26 classes)
| Model | Accuracy | Macro-F1 | Top-3 |
|---|---|---|---|
| Majority class | 0.102 | 0.007 | 0.259 |
| Markov-1 | 0.343 | 0.263 | 0.609 |
| Markov-2 | 0.415 | 0.400 | 0.644 |
| Time-aware Markov | 0.428 | 0.385 | 0.659 |
| **Gradient boosting (temporal features)** | **0.460** | **0.465** | **0.692** |
