# DA331 Assignment 5: CASAS Smart-Home Activity Dataset

| Task | Deliverable |
|---|---|
| 1. EDA notebook | [`01_EDA_CASAS.ipynb`](01_EDA_CASAS.ipynb) |
| 2. Next-activity temporal model | [`02_Next_Activity_Model.ipynb`](02_Next_Activity_Model.ipynb) |
| 3. 1–2 page report | [`REPORT.md`](REPORT.md) (fill the ‹…› values from the notebooks' *Summary* cells, then export to PDF) |

## Running on Kaggle
1. Create a new notebook: **File → Import Notebook** and upload `01_EDA_CASAS.ipynb`.
2. **Add Input** → search for the CASAS dataset (or open it from the dataset page with *New Notebook*).
   The notebook scans everything under `/kaggle/input`, so no path needs editing.
3. **Run All**. Then do the same for `02_Next_Activity_Model.ipynb`.
   Both notebooks are self-contained (no GPU and no internet needed).
4. Save a version with outputs (**Save Version → Save & Run All**), because the assignment asks for notebooks
   *with* outputs. Download the executed `.ipynb` files from the version's output page.

Optional settings in the first code cell:
* `MAX_FILES = 5` to prototype on a few homes first, then `None` for all of them.
* `CASAS_DATA_DIR` environment variable, or a local `data/` folder, to run outside Kaggle.

The loader auto-detects the CASAS text formats (tab/space/CSV, with or without a header, an optional
location column, `.gz`, and old `Activity begin/end` markers), so it works on both the Zenodo release and
the Kaggle mirror.

`tools/build_notebooks.py` regenerates both notebooks from one source (the loader code is shared).
