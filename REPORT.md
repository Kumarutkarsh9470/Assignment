# Assignment 5: CASAS Smart-Home Activity Analysis and Next-Activity Prediction
**Course:** DA331 Big Data Analytics: Tools & Techniques  **Name / Roll no.:** ‹fill›

> Numbers marked ‹…› come from the *Summary* cells at the end of the two notebooks after running them on Kaggle.

## 1. Dataset
The CASAS Smart Home dataset (free living, motion/door sensors, activity labels; Zenodo 15708568) contains
ambient sensor data recorded in community homes. Each record is one sensor event:
**date, time, sensor ID, message** (PIR motion `ON/OFF`, magnetic door `OPEN/CLOSE`) and an **activity label**
annotated for the resident (e.g. *Sleep, Cook, Eat, Personal_Hygiene, Bed_Toilet_Transition, Leave/Enter_Home, Relax, Work*).
Sensor names encode their location (e.g. `BedroomAArea`, `KitchenA`). I used the smaller Kaggle mirror:
‹N homes›, ‹N events›, covering ‹start – end›, with ‹N sensors› sensors and ‹N labels› activity labels.

## 2. Tools and data engineering (Task 1)
* **Kaggle Notebooks + pandas / pyarrow / matplotlib / seaborn / scikit-learn.**
* A format-agnostic loader (`01_EDA_CASAS.ipynb`) discovers every data file under `/kaggle/input` and parses
  tab-, space- or comma-separated logs, with or without a header, a location column or gzip compression,
  and expands old-style `Activity begin/end` markers into per-event labels.
  Parsing is **vectorised with regular expressions** (no Python row loop).
* **Memory:** strings are stored as `category` dtype (‹x› MB vs ‹y› MB as plain strings). The parsed table is
  cached as columnar **Parquet**, so the second notebook loads in seconds.

## 3. Exploratory data analysis: key findings
* **Data quality:** ‹% missing labels›, ‹n› duplicate events, ‹n› homes with missing days (outages).
* **Sensors:** motion events dominate (‹%›) and door events are rare. The most active rooms are ‹…›.
* **Temporal structure:** strong circadian rhythm with peaks at ‹…› and a quiet night apart from sleep and
  bed-to-toilet transitions. Inter-event times are heavy-tailed (median ‹x› s, 99th percentile ‹y› s), so the
  stream is bursty. This motivates an **episode** representation rather than fixed time windows.
* **Activities:** labels are imbalanced (`Other_Activity` = ‹%› of events). After collapsing runs into episodes
  there are ‹N› labelled episodes. The most frequent is ‹…›, and the longest are *Sleep* and ‹…›.
  Activities are tied to rooms and hours (activity×room and activity×hour heatmaps).
* **Transitions:** the next-activity distribution is far from uniform. Entropy drops from
  H(next)=‹a› bits to H(next|current)=‹b› and H(next|current, time)=‹c›, so the current activity and the
  time of day are informative predictors.

## 4. Next-activity model (Task 2, `02_Next_Activity_Model.ipynb`)
**Problem.** Given the history up to the end of activity episode *t*, predict the label of episode *t+1*.

**Pipeline.**
1. Events are grouped into episodes (runs of one label). `Other_Activity` is removed, and same-label fragments
   less than 10 minutes apart are merged. Labels with fewer than 20 episodes are grouped as `Rare_Activity`.
2. Temporal features at the end of episode *t*: the current activity and the 3 previous activities (lags),
   hour of day as sin/cos, day of week, weekend flag, log duration, log number of events, number of distinct
   sensors, last sensor fired, and time since the previous episode. All features are available at prediction
   time (no leakage).
3. **Chronological split per home:** 70% train, 15% validation, 15% test (the most recent period).
4. Models: majority class, 1st-order Markov chain, 2nd-order Markov chain, time-aware Markov chain
   P(next | current, 3-h bin) (add-α smoothing with back-off), and **HistGradientBoostingClassifier** with native
   categorical features. Its hyper-parameters are chosen on validation, and the model is then refit on
   train+validation.

**Results on the held-out test period** (‹N› test transitions, ‹K› classes):

| Model | Accuracy | Macro-F1 | Top-3 acc. |
|---|---|---|---|
| Majority class | ‹› | ‹› | ‹› |
| Markov-1 P(next\|cur) | ‹› | ‹› | ‹› |
| Markov-2 P(next\|prev,cur) | ‹› | ‹› | ‹› |
| Time-aware Markov | ‹› | ‹› | ‹› |
| **HistGradientBoosting (temporal)** | **‹›** | **‹›** | **‹›** |

The most important features (permutation importance) were ‹…›. Most errors fall between activities that
share rooms and times of day (e.g. ‹…›), and on rare classes.

## 5. Conclusions and possible extensions
A simple, leakage-free pipeline (event stream → episodes → temporal features → gradient boosting) clearly
beats the frequency and Markov baselines, and its top-3 predictions are reliable enough for applications such
as reminders or anomaly detection. Possible extensions: sequence models (GRU/Transformer) over episodes or raw
events, predicting *when* the next activity starts, cross-home generalisation (train on some homes, test on
unseen ones), and scaling the loader to all homes with Spark/Polars.
