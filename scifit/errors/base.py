
class GenericError(Exception):
    pass


class MissingModel(GenericError):
    pass


class InputDataError(GenericError):
    pass

