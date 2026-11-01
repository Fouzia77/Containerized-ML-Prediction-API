# Model artifact

The Docker build runs `python scripts/create_model.py` and writes `my_classifier.h5` here. For local execution, install the requirements and run the same script before starting the API. This creates a valid **untrained demo model** so the inference and deployment pipeline can be tested end-to-end. Replace it with a genuinely trained classifier for meaningful predictions.
