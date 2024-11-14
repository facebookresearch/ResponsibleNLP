## SMART Filtering

To perform SMART filtering on a dataset, run the following command:

```
python main.py --dataset <dataset_name>
```

Replace <dataset_name> with the name of your dataset. The dataset must have a corresponding configuration file, e.g., `config_mmlu.py`.

## How it works

The `main.py` script calls the following filtering scripts:
- `pre_filtering.py`: Removes anomalous subsets and exact duplicates from the dataset 
- `filtering_easy_hard.py`: Identifies and removes easy questions from the dataset.
- `filtering_data_contamination.py`: Identifies and removes data-contaminated questions from the dataset.
- `filtering_similar_examples.py`: Identifies and removes similar questions from the dataset.

These scripts work together to refine the dataset and remove unwanted data points. By running main.py, you can apply all of these filters in one step.

## Customization

All customizations can be done in the `config_{dataset_name}.py` file. This file allows you to tailor the filtering process to your specific dataset needs.
