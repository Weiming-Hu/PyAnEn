# PyAnEn

_A standalone module that supports Parallel Analog Ensemble and various verification__

## Please Read This

This is the Python interface and tools for [Parallel Analog Ensemble](https://weiming-hu.github.io/AnalogsEnsemble/). Currently, it is in its early development stage. I encourage you to reach out to [me](https://weiming-hu.github.io/) if you plan to use it. I will be happy to assist you on integrating [Analog Ensemble](https://weiming-hu.github.io/AnalogsEnsemble/) into your scientific workflow.


## Install

```
pip install git+https://github.com/Weiming-Hu/PyAnEn.git
```

## Look-Up Table for Included Verification Metrics

| **Metric**               | **Method Name**     | **Operate Along Axis** | **Support Bootstraping** |
|--------------------------|---------------------|------------------------|--------------------------|
| CRPS                     | crps                | Yes                    | Yes                      |
| Bias                     | error               | Yes                    | Yes                      |
| Spread                   | spread              | Yes                    | Yes                      |
| RMSE                     | sq_error            | Yes                    | Yes                      |
| MAE                      | ab_error            | Yes                    | Yes                      |
| Brier Score              | brier               | Yes                    | Yes                      |
| Spread Skill Correlation | binned_spread_skill | Yes                    | Yes                      |
| Rank Histogram           | rank_hist           | Yes                    | No                       |
| Sharpness                | sharpness           | Yes                    | No                       |
| Reliability Diagram      | reliability         | No                     | No                       |
| ROC Curve                | roc                 | No                     | No                       |
