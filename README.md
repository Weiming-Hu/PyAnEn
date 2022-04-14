# PyAnEn

_A standalone module that supports Parallel Analog Ensemble and various verification__

This project started off as a Python tool for reading and analyzing [Parallel Analog Ensemble](https://weiming-hu.github.io/AnalogsEnsemble/). However, as I continue to work on this, it has evolved to the point of including many more functionalities. Aside from supporting analysis of PAnEn forecasts, it includes many verification functions. The classes can easily be extended to other types of distributions that are currently not included.

I will be happy to assist you to integrate [Analog Ensemble](https://weiming-hu.github.io/AnalogsEnsemble/) into your scientific workflow.

Documentation is currently being written.

## Install

```
pip install git+https://github.com/Weiming-Hu/PyAnEn.git
```

## Implemented Verification Metrics

| **Metric**               | **Method Name**     | **Operate Along Axis** | **Support Bootstraping** | **Parallelizable** |
|--------------------------|---------------------|------------------------|--------------------------|--------------------|
| CRPS                     | crps                | Yes                    | Yes                      | No                 |
| Bias                     | error               | Yes                    | Yes                      | No                 |
| Spread                   | spread              | Yes                    | Yes                      | No                 |
| RMSE                     | sq_error            | Yes                    | Yes                      | No                 |
| MAE                      | ab_error            | Yes                    | Yes                      | No                 |
| Correlation              | corr                | Yes                    | Yes                      | No                 |
| Brier Score              | brier               | Yes                    | Yes                      | Yes for ensembles  |
| Spread Skill Correlation | binned_spread_skill | Yes                    | Yes                      | No                 |
| IOU (Deterministic)      | iou_determ          | Yes                    | No                       | No                 |
| IOU (Probability)        | iou_prob            | Yes                    | No                       | No                 |
| Rank Histogram           | rank_hist           | Yes                    | No                       | Yes                |
| Sharpness                | sharpness           | Yes                    | No                       | Yes for ensembles  |
| Reliability Diagram      | reliability         | No                     | Yes                      | Yes for ensembles  |
| ROC Curve                | roc                 | No                     | No                       | Yes for ensembles  |

Some operations are not parallelized currently because these operations are already blazingly fast. However, parallelization can be done if needed.
