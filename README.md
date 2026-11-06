# NuLKT code

Training and evaluation code for NuLKT second-language error prediction.

Use Python 3.12. From this folder, install the dependencies:

```sh
python -m pip install -r requirements.txt
```

Obtain the Duolingo SLAM English-learning track (dataset DOI
10.7910/DVN/8SWHNO) and place the extracted train/dev/test and key files in:

```text
data/raw/extracted/data_en_es/
```

Prepare the data and run the three fixed full-pipeline repetitions:

```sh
python -X utf8 scripts/reproduce_offline_safe.py --prepare
python -X utf8 reviewer_experiments/run_prespecified_repeats.py --source . --output ../NuLKT_runs
python -X utf8 reviewer_experiments/summarize_completed_repeats.py --runs ../NuLKT_runs --out ../NuLKT_summary
```

For individual training/evaluation commands:

```sh
python scripts/reproduce_offline_safe.py --help
```

The tests are run with `python -m pytest tests -q`. Long training requires
substantial computing time. The published-test evaluations are post hoc.
The MIT license applies to code; dataset terms remain separate.
