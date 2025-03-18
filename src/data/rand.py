import random  # Import the random module for generating random numbers

class Uniform(object):
    """
    A class to generate random numbers from a uniform distribution.
    
    This class provides a method to sample random numbers uniformly distributed
    between a specified range [a, b].
    """
    def __init__(self, a, b):
        """
        Initialize the Uniform distribution sampler.
        
        Args:
            a (float): The lower bound of the uniform distribution.
            b (float): The upper bound of the uniform distribution.
        """
        self.a = a  # Store the lower bound
        self.b = b  # Store the upper bound

    def sample(self):
        """
        Sample a random number from the uniform distribution.
        
        Returns:
            float: A random number uniformly distributed between [a, b].
        """
        return random.uniform(self.a, self.b)  # Generate and return a uniform random number


class Gaussian(object):
    """
    A class to generate random numbers from a Gaussian (normal) distribution.
    
    This class provides a method to sample random numbers from a Gaussian
    distribution with a specified mean and standard deviation.
    """
    def __init__(self, mean, std):
        """
        Initialize the Gaussian distribution sampler.
        
        Args:
            mean (float): The mean (center) of the Gaussian distribution.
            std (float): The standard deviation (spread) of the Gaussian distribution.
        """
        self.mean = mean  # Store the mean of the distribution
        self.std = std  # Store the standard deviation of the distribution

    def sample(self):
        """
        Sample a random number from the Gaussian distribution.
        
        Returns:
            float: A random number sampled from the Gaussian distribution.
        """
        return random.gauss(self.mean, self.std)  # Generate and return a Gaussian random number


class Constant(object):
    """
    A class to return a constant value.
    
    This class provides a method to always return the same constant value,
    regardless of how many times it is sampled.
    """
    def __init__(self, val):
        """
        Initialize the Constant value sampler.
        
        Args:
            val (float): The constant value to return.
        """
        self.val = val  # Store the constant value

    def sample(self):
        """
        Return the constant value.
        
        Returns:
            float: The constant value provided during initialization.
        """
        return self.val  # Return the constant value