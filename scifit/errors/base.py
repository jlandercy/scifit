
class GenericError(Exception):
    pass


class ConfigurationError(GenericError):
    pass


class MissingModel(GenericError):
    pass


class InputDataError(GenericError):
    pass


class NotFittedError(GenericError):
    pass


class NotSolvedError(GenericError):
    pass

