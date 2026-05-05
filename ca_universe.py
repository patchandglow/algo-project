"""Base class for all strategies."""

from abc import ABC, abstractmethod


class Strategy(ABC):
    name = "base"

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def generate_signals(self, data):
        """Return DataFrame with at minimum a 'signal' column.
        +1 = long, 0 = flat, -1 = short. Signal at row N = position at close of N.
        """
        ...

    def __repr__(self):
        return f"{self.name}({self.params})"
