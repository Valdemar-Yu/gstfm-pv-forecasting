# Vendored models from thuml/Time-Series-Library (MIT License).
# See models/baselines/LICENSE-TSLIB for the full license text.
# Model modules (DLinear, TimesNet, FEDformer) are imported lazily by
# models/baselines/wrappers.py so that optional dependencies of one model
# (e.g. sympy/scipy for FEDformer's wavelet layers) do not break the others.
