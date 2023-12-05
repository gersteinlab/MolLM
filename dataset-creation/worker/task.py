from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic

T = TypeVar("T")


class Task(ABC, Generic[T]):
    @abstractmethod
    def get_batch(self, batch_size: int) -> List[T]:
        pass

    @abstractmethod
    def process(self, obj: T):
        pass

    @abstractmethod
    def mark_complete(self, obj: T):
        pass