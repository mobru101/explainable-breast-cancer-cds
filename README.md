# Explainable Clinical Decision Support for Breast Cancer Diagnosis

This repository contains the full implementation of an explainable machine learning–based clinical decision support prototype for breast cancer diagnosis using numerical cell morphology features.

The project combines predictive modeling, post-hoc explainability, clinician workflow insights and an interactive Streamlit-based interface to demonstrate how explainable AI methods can be translated into practical healthcare applications.

## Project Overview

Breast cancer diagnosis based on cell morphology is a well-studied machine learning task with strong predictive performance. However, clinical adoption remains limited due to:
- lack of model transparency,
- insufficient integration into clinical workflows,
- and limited support for clinical reasoning in ambiguous cases.

This project addresses these challenges by:
- evaluating multiple machine learning models,
- integrating SHAP-based explainability,
- incorporating feedback from practicing pathologists,
- and implementing an interactive clinician-facing prototype.

## Dataset
- Breast Cancer Wisconsin (Diagnostic) Dataset
- 569 samples, 30 numerical features derived from cell nucleus morphology
- Binary classification: benign vs. malignant

Source: Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.

## Methods

Predictive Models

The following models were implemented and evaluated:
- Logistic Regression (interpretable baseline)
- Support Vector Machine (linear)
- Random Forest
- XGBoost (final selected model)

Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC–AUC
- Confusion Matrix
- Learning Curves (overfitting assessment)

Explainability:
- SHapley Additive exPlanations (SHAP)
- Global feature importance
- Local, case-specific explanations
- TreeExplainer for exact SHAP computation on tree-based models

## Clinical Decision Support Prototype

An interactive prototype was implemented using Streamlit, allowing users to:
- input numerical cell morphology features,
- receive a malignancy risk assessment,
- explore SHAP-based explanations,
- view results in a structured dashboard designed for clinical interpretation.

The prototype demonstrates how explainable models can be embedded into clinician-facing workflows.

## Disclaimer

This project is intended for research and educational purposes only.
It is not a certified medical device and must not be used for clinical decision-making without proper validation and regulatory approval.
