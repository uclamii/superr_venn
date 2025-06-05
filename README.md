# Usage

See ./notebooks/examples.ipynb for example usage.

Functions:
```
create_error_profile(df,
                    id,
                    columns,
                    set_annotations,
                    incorrect_value,
                    figsize)

create_oddratio_profile(df,
                    subgroup,
                    id,
                    columns,
                    set_annotations,
                    incorrect_value,
                    figsize)

create_stratified_error_profile(df,
                    subgroup,
                    id,
                    columns,
                    set_annotations,
                    incorrect_value,
                    figsize)
```

- df: Dataframe where columns represent correct/incorrect predictions from different models
- id: column representing identifiers for each row/sample
- columns: which columns represent models
- set_annotations: names of each model
- incorrect_value: value in column representative of an error
- subgroup: additional categorical column stratifying samples into subpopulations

## Example 
Below is an example dataframe.

```
display(df)
```
|    |   Model 1 |   Model 2 |   Model 3 |   Model 4 |   Model 5 |   Patient ID | Subgroup   |
|---:|----------:|----------:|----------:|----------:|----------:|-------------:|:-----------|
|  0 |         1 |         0 |         1 |         1 |         1 |            1 | A          |
|  1 |         0 |         1 |         0 |         1 |         0 |            2 | B          |
|  2 |         0 |         0 |         0 |         0 |         0 |            3 | B          |
|  3 |         0 |         0 |         1 |         0 |         0 |            4 | A          |
|  4 |         0 |         0 |         0 |         0 |         0 |            5 | A          |

```
create_error_profile(
    df,
    id="Patient ID",
    columns=[f"Model {i + 1}" for i in range(models)],
    incorrect_value=1,
)
```
![alt text](image.png)

# Credit
Source repository: https://github.com/gecko984/supervenn. Most of the algorithms are adapted from the original repo. The edits are to extend the visualizations for error analysis.
