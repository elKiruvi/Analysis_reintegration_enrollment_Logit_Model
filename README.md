# Determinants of Reintegration Program Enrollment in Colombia

## Overview

This repository contains the R code for an econometric analysis aimed at identifying the key factors that influence whether a demobilized person from an armed group in Colombia enrolls in the official reintegration program.

The primary research question is: **What socio-economic and demographic characteristics are significant predictors of program enrollment?**

The project uses a **binomial logistic regression (logit) model** to estimate the probability of enrollment based on a set of individual characteristics.

## Methodology

The analysis follows these key steps:
1.  **Data Cleaning & Preparation:** Loading the raw data and transforming categorical variables into dummies using `dplyr`.
2.  **Model Estimation:** Fitting an initial logit model with all potential predictors.
3.  **Model Selection:** Using the Akaike Information Criterion (AIC) via a stepwise procedure (`stepAIC`) to select a more parsimonious and interpretable model.
4.  **Model Evaluation & Diagnostics:**
    *   Assessing model fit with McFadden's Pseudo R-squared.
    *   Evaluating predictive power with a confusion matrix and the Area Under the ROC Curve (AUC).
    *   Ensuring robustness by calculating Heteroskedasticity-Consistent (HC) standard errors.
5.  **Visualization:** Creating plots with `ggplot2` to illustrate key findings.

## How to Run

1.  Clone or download this repository.
2.  Open the R project in RStudio.
3.  Ensure you have the required packages installed (`readr`, `dplyr`, `MASS`, `pROC`, `lmtest`, `sandwich`, `ggplot2`).
4.  Run the `analysis_reintegration_enrollment.R` script to replicate the analysis.
