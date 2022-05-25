# AB-Hypothesis-Testing

<!-- ![Wordcloud](assets/CH.PNG?raw=true "workflow") -->

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

## Contrbutors
- Yididiya Samuel

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


## License
[MIT](https://choosealicense.com/licenses/mit/)
