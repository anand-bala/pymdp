from abc import ABC, abstractmethod
from typing import Generic, Iterable, Protocol, TypeVar, Union

T = TypeVar("T")


class Distribution(Protocol[T]):
    def sample(self) -> T:
        """Returns a random element from the distribution."""
        ...

    def pdf(self, x: T) -> float:
        """Evaluates the probability density of the distribution at sample `x`."""
        ...

    def mean(self) -> float:
        """Returns the mean of the distribution."""
        ...

    def mode(self) -> float:
        """Returns the most likely value in the distribution."""
        ...

    def support(self) -> Iterable[T]:
        """Returns the support of the distribution

        Returns:
            An iterable object containing the possible values that can be sampled from
            the distribution. Values with zero probability may be skipped.

        """
        ...


S = TypeVar("S")
A = TypeVar("A")


class MDP(Generic[S, A], ABC):
    """
    Abstract base type for a Markov decision process.
    """

    @property
    @abstractmethod
    def discount(self) -> float:
        ...

    @abstractmethod
    def transition(self, state: S, action: A) -> Distribution[S]:
        """Returns the transition distribution from the current state-action pair."""
        ...

    @abstractmethod
    def reward(self, s: S, a: A, sp: S) -> float:
        """Returns the immediate reward for the `(s, a, s')` tuple/transition."""
        ...

    @abstractmethod
    def isterminal(self, s: S) -> bool:
        """Checks if state `s` is terminal."""

    @property
    @abstractmethod
    def initialstate(self) -> Union[S, Distribution[S]]:
        """Return a distribution of initial states (or a single initial state) in the MDP."""
        ...

    @abstractmethod
    def stateindex(self, s: S) -> int:
        """Returns the integer index of state `s`. Used for discrete models only."""
        ...

    @abstractmethod
    def actionindex(self, s: S) -> int:
        """Returns the integer index of action `s`. Used for discrete models only."""
        ...
