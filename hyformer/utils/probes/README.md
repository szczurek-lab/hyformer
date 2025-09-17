# Probes

Public API surface (import paths in parentheses):

- RFProbe (`hyformer.utils.probes.rf.RFProbe`)
- KNNProbe (`hyformer.utils.probes.knn.KNNProbe`)
- MLPProbe (`hyformer.utils.probes.mlp.MLPProbe`)

All probes support both tasks via `task_type`:
- `task_type='classification'`
- `task_type='regression'`

Under the hood, each probe selects an appropriate estimator and uses task-aware CV and scoring via the shared `ProbeBase`.

## Quickstart

Classification (binary):
```python
import numpy as np
from hyformer.utils.probes.rf import RFProbe

X = np.random.randn(1000, 256).astype('float32')
y = (np.random.rand(1000) > 0.5).astype('int64')  # 0/1

probe = RFProbe(task_type='classification')
probe.fit(X, y, selection_metric='auprc')
metrics = probe.evaluate(X, y, plot_curves=False)
print(metrics)
```

Regression:
```python
import numpy as np
from hyformer.utils.probes.knn import KNNProbe

X = np.random.randn(500, 128).astype('float32')
y = np.random.randn(500).astype('float32')

probe = KNNProbe(task_type='regression')
probe.fit(X, y, selection_metric='rmse')
metrics = probe.evaluate(X, y, plot_curves=False)
print(metrics)
```

MLP probe:
```python
from hyformer.utils.probes.mlp import MLPProbe
probe = MLPProbe(task_type='classification')  # or 'regression'
probe.fit(X, y, selection_metric='auroc')
```

## Scoring and CV

- Classification: StratifiedKFold; scoring supports `auprc`, `auroc`, `accuracy`, `f1`.
- Regression: KFold; scoring supports `rmse` (`neg_root_mean_squared_error`), `r2`.

## Data expectations

- `X`: numpy array of shape `(num_samples, feature_dim)`, dtype float.
- `y`:
  - classification: binary 0/1 int array
  - regression: float array

## Accessing results

```python
probe.best_estimator_   # trained sklearn estimator
probe.get_best_params() # dict of best hyperparameters
probe.get_cv_results()  # GridSearchCV results dict
```


