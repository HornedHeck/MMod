import abc
from abc import ABC


class BaseRandomValue(ABC):

    @abc.abstractmethod
    def next(self) -> int:
        pass
