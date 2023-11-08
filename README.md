# CANSSI-Horse-Racing

This repository is dedicated for participation in French Trot Horse Racing: Forecasting Competition organized by CANSSI Ontario.

We are undergraduate students with diverse background which include Statistics, Chemistry and Computer Science from University of Toronto.

## How to test the Model

The model was built using Python 3.10 version

Run the following commands to install necessary modules

```console
pip install -U scikit-learn
pip install pandas
pip install numpy
```
Place the data file name at the following code location under load_data() function

```
data = pd.read_parquet('file_name.parquet', engine='fastparquet')
```

Run the main function. 

A parquet file will be generated in the same folder.

## Contributors

- Minghao Guo (minghaokg@gmail.com)
- Seri Ban (serena.ban@mail.utoronto.ca)
- Seo Won Yi (sean.yi@mail.utoronto.ca)

## About Explanatory Notebook

We did not anticipate to submit the explanatory notebook. Since we didn't have enough time to compile all our pre-processing and analysis codes together neatly, we instead created a pdf document which briefly explains our thought process of what and how we came up with the final model. Rough works from individuals can be found in Rough Draft folder.
