# AB-Hypothesis-Testing

<!-- Table of contents -->
- [AB-Hypothesis-Testing](#ab-hypothesis-testing)
  - [About](#about)
  - [Objectives](#objectives)
  - [Data](#data)
  - [Repository overview](#repository-overview)
  - [Requirements](#requirements)
  - [Usage](#usage)
    - [Option 1: Docker Image](#option-1-docker-image)
    - [Options 2: Local Development](#options-2-local-development)
  - [Contrbutors](#contrbutors)
  - [Contributing](#contributing)
  - [License](#license)

## About
The finance team at Rossman Pharmaceuticals wants to forecast sales in all their stores across several cities six weeks ahead of time. Managers in individual stores rely on their years of experience as well as their personal judgment to forecast sales. 

The data team identified factors such as promotions, competition, school and state holidays, seasonality, and locality as necessary for predicting the sales across the various stores.

The task was to build and serve an end-to-end product that delivers this prediction to analysts in the finance team. 


## Objectives
To build a predictive model that can be used to forecast sales in all stores across all cities 6 weeks ahead.

## Data
The data used for this project is a subset of the [Rossman Pharmaceuticals Sales Data](https://www.kaggle.com/c/rossmann-store-sales/data) dataset.

## Repository overview
 Structure of the repository:
 
        ├── models  (contains trained model)
        ├── .github  (github workflows for CI/CD, CML)
        ├── screenshots  (Important screenshots)
        ├── mlruns  (contains MLflow runs)
        ├── train (contains training scripts) 
        ├── assets  (contains assets)
        ├── data    (contains data of train, store, and test)
        ├── scripts (contains the main script)	
        │   ├── logger.py (logger for the project)
        overview)
        │   ├── plot.py (handles plots)
        │   ├── preprocess.py (Data preprocessing)
        ├── notebooks	
        │   ├── sales-eda.ipynb (overview of the sales Data)
        │   ├── preprocess.ipynb (Preparing the data)
        │   ├── model.ipynb (regression model)
        ├── tests 
        │   ├── test_preprocess.py (test for the the AB testing script)
        ├── README.md (contains the project description)
        ├── requirements.txt (contains the required packages)
        |── LICENSE (license of the project)
        ├── setup.py (contains the setup of the project)
        └── .dvc (contains the dvc configuration)

## Requirements
The project requires the following:
python3
Pip3

## Usage
### Option 1: Docker Image
The docker image is built on docker-hub on every push to the main branch using Github actions. It can be used to run the project locally.
Pull docker image
```
docker pull jedisam/sales-forecasting
```
Run docker image
```
docker run --rm -it  -p 8501:8501/tcp jedisam/sales-forecasting
```
### Options 2: Local Development
**1.Activate environement or create one:**
```
conda create -n <env-name> && conda activate <env-name>
```
**2.Install required packages**
```
pip install -r requirements.txt
```
**3.Run the app**
```
python3 wsgi.py
```
you should be able to see the dashboard\api.



## Contrbutors
- Yididiya Samuel

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


## License
[MIT](https://choosealicense.com/licenses/mit/)
