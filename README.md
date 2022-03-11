# PyAnEn

_A standalone module that supports Parallel Analog Ensemble and various verification__

## Please Read This

This is the Python interface and tools for [Parallel Analog Ensemble](https://weiming-hu.github.io/AnalogsEnsemble/). Currently, it is in its early development stage. I encourage you to reach out to [me](https://weiming-hu.github.io/) if you plan to use it. I will be happy to assist you on integrating [Analog Ensemble](https://weiming-hu.github.io/AnalogsEnsemble/) into your scientific workflow.


## Install

```
pip install git+https://github.com/Weiming-Hu/PyAnEn.git
```

## Look-Up Table for Included Verification Metrics

| **Metric**               | **Method Name**     | **Operate Along Axis** | **Support Bootstraping** | **Parallelizable** |
|--------------------------|---------------------|------------------------|--------------------------|--------------------|
| CRPS                     | crps                | Yes                    | Yes                      | No                 |
| Bias                     | error               | Yes                    | Yes                      | No                 |
| Spread                   | spread              | Yes                    | Yes                      | No                 |
| RMSE                     | sq_error            | Yes                    | Yes                      | No                 |
| MAE                      | ab_error            | Yes                    | Yes                      | No                 |
| Brier Score              | brier               | Yes                    | Yes                      | Yes for ensembles  |
| Spread Skill Correlation | binned_spread_skill | Yes                    | Yes                      | No                 |
| IOU (Deterministic)      | iou_determ          | Yes                    | No                       | No                 |
| IOU (Probability)        | iou_prob            | Yes                    | No                       | No                 |
| Rank Histogram           | rank_hist           | Yes                    | No                       | Yes                |
| Sharpness                | sharpness           | Yes                    | No                       | Yes for ensembles  |
| Reliability Diagram      | reliability         | No                     | No                       | Yes for ensembles  |
| ROC Curve                | roc                 | No                     | No                       | Yes for ensembles  |

Some operations are not parallelized currently because these operations are already blazingly fast. However, parallelization can be done if needed.
