# Vietnamese OCR with CRNN

Text Recognition from images using CRNN for Vietnamese characters.

## Install requirements

```
pip install -r requirements.txt
```
## Dataset

Dataset used for project [here](https://drive.google.com/file/d/1dVO8yyqvyGVeWnQ78C5WYOdjCwaa7mUr/view?usp=sharing).

## Model's weights

Pretrained model for prediction and evaluation [vn_model.h5](https://drive.google.com/file/d/1-WmGlGVQtyrcPFCqwMYiwULKXvO_XM8M/view?usp=sharing)

### Prepare Data

Download dataset and pretrained model from Google Drive. Then, extract and \
split dataset into train, validate and test set.

```
python libs/prepare/prepare_data.py
```

## Run application

```
python main.py
```

Application run on <u>http://localhost:8096 </u>

## Api

1. Training model
    - URL: /train
    - Method: GET
    - Params:
         - **epochs**: Number of training epochs
    - Usage: Training model and save weights

2. Evaluation 
    - URL: /evaluate
    - Method: POST
    - Body:
        - **filename**: Path to file contain image's paths and labels
    - Usage: Evaluate accuracy and letter accuracy of model with data described in filename or (paths, labels).
    - Return: Accuracy, letter accuracy

3. Prediction
    - URL: /predict
    - Method: POST
    - Body:
        - **img**: Path to image for prediction
    - Usage: Predict text in new image
    - Return: Text predicted
