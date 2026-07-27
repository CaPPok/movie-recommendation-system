"""AWS integration: S3 sync, DynamoDB export, and retraining configuration.

Nothing in the rest of the repository imports this package. The data pipeline,
training, evaluation and serving all run with no AWS account and no boto3
installed; that separation is deliberate, so a credential problem can never
break a local run.
"""
