# ENGG2112 – Streamlining the Surge

## Project Overview

This repository contains all code, data, reports, rendered outputs, application files, and presentation materials developed for the ENGG2112 project **Streamlining the Surge**.

The project investigates the use of machine learning and clinical decision-support tools to improve patient triage and resource allocation during hospital surge events.

---

## Repository Structure

### Code Files

Contains all source code used to develop, train, tune, and evaluate the machine learning models:

- Model 1 – Admission Severity Prediction
- Model 1.2 – Rapid-Test Admission Severity Prediction
- Model 2 – COVID Deterioration Prediction
- Model 2.2 – Rapid-Test COVID Deterioration Prediction

This folder contains the Quarto documents, Python scripts, feature selection pipelines, hyperparameter optimisation workflows, threshold tuning procedures, model evaluation code, and report generation files used throughout the project.

---

### data

Contains the datasets used for model development and evaluation.

These are the primary datasets used to train, validate, and test the machine learning models described in the project.

---

### patient_data_mvp

Contains the patient data CSV files used as inputs for the Minimum Viable Product (MVP) application.

These files demonstrate the required input format for generating predictions through the triage application.

---

### HTML Files

Contains pre-rendered HTML versions of:

- Model 1
- Model 1.2
- Model 2
- Model 2.2

These files can be viewed directly in a web browser without requiring Quarto, RStudio, Python, or any additional software.

---

### ENGG2112_Triage_App

Contains the source code for the interactive clinical decision-support application developed as part of the project.

The application integrates the trained machine learning models and provides an interface for patient assessment, risk prediction, and triage decision support.

---

### icu_triage_simulation.py

Contains the surge simulation model used to evaluate triage strategies under increasing hospital demand.

The simulation was developed to investigate resource allocation, patient prioritisation, and healthcare system performance during surge scenarios.

---

### actual presentation

Contains the final presentation materials used for project submission, including:

- Presentation slides
- Supporting presentation assets
- Quarto slide source files
- Rendered HTML presentation files

---

## Models Included

### Model 1

Predicts admission severity using metabolomic and clinical patient data.

### Model 1.2

Rapid-test version of Model 1 using only features readily available at patient presentation.

### Model 2

Predicts future COVID deterioration using metabolomic and clinical information available at a single presentation.

### Model 2.2

Rapid-test version of Model 2 using a reduced feature set suitable for emergency triage settings.

---

## Application Components

The repository includes:

- Machine learning model development and evaluation code
- Interactive triage application source code
- Surge simulation code
- Rendered HTML reports
- Presentation source files and slide decks
- MVP patient input datasets

---

## Technologies Used

- Python
- R
- Quarto
- Scikit-Learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Authors

ENGG2112 – Streamlining the Surge Project Team

The University of Sydney
