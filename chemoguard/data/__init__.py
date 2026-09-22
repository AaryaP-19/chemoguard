"""Dataset ingestion, with units declared at the source boundary.

Every loader in this package states the unit of every column it reads, in
source, before any value is used. A dataset with no declared units does not
load -- the same rule ``train_risk_prediction_models.SOURCE_UNITS`` applies to
training (CLAUDE.md Finding 7).
"""
