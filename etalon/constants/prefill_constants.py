# Prefill lengths profile over, all powers of 2 between 256 and 128K
PREFILL_VALUES = [2**i for i in range(8, 15)]
# Model to train on the prefill values and prefill times
PREFILL_MODEL = "RandomForestRegressor"
# Random Forest Regressor parameters
PREFILL_RANDOM_FOREST_PARAMS = {
    "n_estimators": 10,
    "random_state": 0,
}
# Polynomial degree for the prefill time predictor
PREFILL_POLYNOMIAL_DEGREE = 2
# RMSE threshold for the prefill time predictor
PREFILL_RMSE_THRESHOLD = 0.05
# Number of Ray clients to use for prefill profiling
PREFILL_NUM_CLIENTS = 1
# Number of concurrent requests per client for prefill profiling
PREFILL_NUM_CONCURRENT_REQUESTS_PER_CLIENT = 1
# Number of completed requests to wait for before stopping the prefill profiling for a prompt length
PREFILL_MAX_NUM_COMPLETED_REQUESTS = 1
