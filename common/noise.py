import numpy as np

def parametric_uncertainty(parameters, uncertainty, RNG):
    """
    Adds uncertainty to a vector of parameters based on the uncertainty parameter.
    
    Args:
        parameters (np.ndarray): The original parameter vector.
        uncertainty_param (float): The level of uncertainty to add.
        RNG: random number generator.

    Returns:
        np.ndarray: Parameter vector with added uncertainty.
    """
    # The following crop parameters in the parameter vector are perturbed
    parameters = np.array(parameters)
    noise = RNG.uniform(-uncertainty, uncertainty, size=parameters.shape[0])
    parameters += noise*parameters
    return parameters

if __name__ == "__main__":
    # Example usage
    params = np.array([1.0, 2.0, 3.0, 4.0])
    uncertainty_level = 0.1  # 10% uncertainty
    rng = np.random.default_rng(666)  # Set seed for reproducibility

    noisy_params = parametric_uncertainty(params, uncertainty_level, rng)
    print("Original parameters:", params)
    print("Parameters with noise:", noisy_params)
