<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

# Substantive Fairness in Conformal Prediction

This repository contains the codebase for experiments studying **procedural and substantive fairness properties of conformal prediction methods**, including downstream evaluation and LLM-in-the-loop analysis. Experiments using this code are presented in  [*Beyond Procedure: Substantive Fairness in Conformal Prediction*](https://arxiv.org/abs/2602.16794).

The code supports multiple datasets, score functions, and evaluation pipelines through a unified, configuration-driven interface.

---

## Environment Setup

This project uses **`uv`** for Python dependency management.

### 1. Install `uv`

If not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create the environment

From the repository root:

```bash
uv sync
```

This will create and manage a local virtual environment with all required dependencies.

To run commands inside the environment:

```bash
uv run main.py
```
or

```bash
uv run env PYTHONPATH=src python main.py
```

---

## Datasets

The following datasets are supported:

* `bios`
* `ravdess`
* `facet`
* `acs-income`

Create a top-level data directory:

```bash
mkdir data
```

### BiosBias (`bios`)

We use a preprocessed version of the BiosBias dataset, encoded using a BERT-base model.

1. Create the directory:

   ```bash
   mkdir -p data/BiosBias
   ```
2. Download the preprocessed `.pickle` files from this [Google Drive](https://drive.google.com/drive/folders/1TW6lFZCxuUPzy3A42_MSfHEWSwRP9zYP?usp=drive_link).
3. Place them in:

   ```
   data/BiosBias/
   ```

---

### RAVDESS (`ravdess`)

1. Download the dataset from:

   ```
   https://zenodo.org/records/1188976
   ```
2. Unzip and place the contents in:

   ```
   data/RAVDESS/
   ```

---

### FACET (`facet`)

1. Download the dataset from:

   ```
   https://facet.metademolab.com/
   ```
2. Unzip and place it in:

   ```
   data/facet/
   ```
3. Ensure the following structure:

   ```
   data/facet/
   ├── images/
   │   ├── imgs_1/
   │   ├── imgs_2/
   │   └── imgs_3/
   └── annotations/
   ```

---

### ACS Income (`acs-income`)

The ACS Income dataset is downloaded and processed automatically.

No manual download is required. Dataset setup is handled by:

```
src/internal/dataset/datasets/acs_downloader.py
```
Run

```bash
uv run python src/internal/dataset/datasets/acs_downloader.py
```
to download the ACS Income data to the data/acs folder.

---

## Configuration System

This project uses **Hydra** for configuration management.

Configurations are composed from:

1. A base config:

   ```
   src/substantive/faircp/conf/config.yaml
   ```
2. A dataset-specific config:

   ```
   src/internal/conf/dataset/<dataset>.yaml
   ```
3. A custom config:

   ```
   custom_config.yaml
   ```
4. Optional runtime overrides via Hydra

At runtime, these are merged and validated automatically.

---

## Usage

### Main Entry Point

The main entry point is:

```bash
uv run main.py
```

Running this command launches an **interactive menu**:

```
📊  Conformal Prediction - Fairness Project Runner
==================================================
[1] Run Conformal
[2] Run LLM-in-loop
[3] Chart and Heatmap
[4] Compute statistics results
```

You will then be prompted to select a dataset.

---

### Pipeline Options

**1. Run Conformal**

* Loads dataset
* Trains or loads a base model
* Performs conformal calibration
* Produces prediction sets
* After successfully running, outputs will be in a timestamped folder under `logs/`

**2. Run LLM-in-the-loop**

* Ensure your API key is exported, e.g. `export OPENAI_API_KEY="<api_key>"`
* Specify the timestamped folder from stage 1 as `conformal_result_dataset` in `custom_config.yaml`
* Alternatively, add a flag to your `uv run` command with the output folder with results from stage 1
`uv run main.py conformal_result_dataset=<timestamped_folder>`
* Runs downstream evaluation with LLM feedback
* Uses conformal outputs as inputs to the LLM

**3. Chart and Heatmap**

* Specify the timestamped folder from stage 1 as `conformal_result_dataset` in `custom_config.yaml`
* Uses conformal outputs to create plots
* Generates visual diagnostics (coverage, gap metrics, heatmaps)

**4. Compute Statistics**

* Specify the timestamped folder from stage 2 as `statistics_result_dataset` in `custom_config.yaml`
* Specify the dataset of interest (`ravdess`, `bios`, `facet`, `acs-income`) as `statistics_dataset` in `custom_config.yaml`
* Uses LLM prediction outputs to calculate metrics and statistics, including GEE analysis  
* Computes aggregate fairness and performance metrics across runs

---

## Adding or Modifying Datasets

To add a new dataset:

1. Implement the dataset class under:

   ```
   src/internal/dataset/
   ```
2. Add a dataset config file:

   ```
   src/internal/conf/dataset/<name>.yaml
   ```
3. Register the dataset in `DATASET_CLASS_MAP`

---

## Outputs

Depending on the selected pipeline, outputs may include:

* Serialized configs
* Prediction sets
* Fairness metrics
* Aggregated statistics
* Visualization artifacts

Output locations are controlled by configuration files.

---

## Citing

If you use any part of this repository in your research, please cite the associated paper with the following bibtex entry:

```
@article{liu2026beyondprocedure,
  title={Beyond Procedure: Substantive Fairness in Conformal Prediction},
  author={Liu, Pengqi and Yu, Zijun and Belbahri, Mouloud and Charpentier, Arthur and Asgharian, Masoud and Cresswell, Jesse},
  journal={arXiv:2602.16794},
  year={2026}
}
```

## License

This data and code is licensed under the MIT License, copyright by Layer 6 AI.
