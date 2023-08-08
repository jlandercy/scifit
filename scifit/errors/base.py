class GenericError(Exception):
    """
    Generic error all other errors inherit from.
    """

    pass


class ConfigurationError(GenericError):
    pass


class MissingModel(GenericError):
    """
    Error raised when model is not defined
    """

    pass


class InputDataError(GenericError):
    """
    Error raised when input data validation fails
    """

    pass


class NotStoredError(GenericError):
    pass


class NotFittedError(GenericError):
    pass


class NotSolvedError(GenericError):
    pass
