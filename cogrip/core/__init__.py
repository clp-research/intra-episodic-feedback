import abc


class Jsonable(abc.ABC):

    def to_json(self):
        raise NotImplementedError()
