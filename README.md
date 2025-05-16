# College Ranking & Score Prediction API (ML-Powered)
Visit Here: https://nirf-rank-predictor.netlify.app/ 
## Project Description

This project implements a RESTful API using Flask to provide predictions for key metrics related to college assessment, including SS Score, GSM, and potentially aspects influencing NIRF Ranking, utilizing Machine Learning models trained on relevant data.

The primary goal is to offer a programmatic way to estimate these scores based on input parameters, which can be useful for internal analysis, comparative studies, or integration into other applications.

## Features

* API endpoint for predicting SS Score based on input parameters.
* Uses a trained Polynomial Regression model for SS and GSM score prediction.
* Includes data validation and basic error handling for API requests.
* NIRF Rank prediction based on common metrics such as TLR, RPC, GO etc.
* Poisson regressor used for predicting NIRF Ranks.


## Technologies Used

* Python
* Flask
* Flask-CORS
* Pandas
* NumPy
* Scikit-learn (sklearn)

## Prerequisites for locally running it/source code analysis

Before you begin, ensure you have the following installed:

* Python 3.6+
* pip (Python package installer)


