class GenericError(Exception):
    """
    Generic error all other errors inherit from.
    """
    pass


class ConfigurationError(GenericError):
    pass


class MissingModel(GenericError):
    pass


class InputDataError(GenericError):
    """
    Error raised when input data validation fails
    """
    pass


class NotFittedError(GenericError):
    pass


class NotSolvedError(GenericError):
    pass
