## Dataset Structure

- Unzip `dataset.zip` file to get `dataset.db`
- `dataset.db` file contains only samples for training the classifier (readable/non-readable). Each row has `url` string, initial `html` string and the corresponding `class` number.
- We use binary classification; Readable and non-readable samples belong to class 1 and 0 respectively. 
- URLs and their corresponding labels are available in `/labels/labels.csv`
- Labels are explained in `/labels/labels-legend.txt` 
- D1, D2, and D3 refer to article, random, and landing pages (more details can be found in the paper)

## Classifier Performance Result

- Get all the dependencies (running within a separate `virtualenv` is recommended):

    ```pip install -r requirements.txt```
- Evaluate the model:

     ```python model.py --dbname "/path/to/dataset.db" --threads [# of threads]```
- Once it's done, the classification report will be printed on the screen

Tested on Ubuntu 16.04.5 LTS and macOS Mojave 10.14.3

## Note

- Everything in this repository is written in Python 2.7

