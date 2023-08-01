
class GenericError(Exception):
    pass


class MissingModel(GenericError, NotImplemented):
    pass


class InputDataError(GenericError, TypeError):
    pass

