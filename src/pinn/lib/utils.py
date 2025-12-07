from collections.abc import Callable, Iterable
from typing import TypeVar, cast

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

T = TypeVar("T")


def find(
    iterable: Iterable[T],
    predicate: Callable[[T], bool],
    default: T | None = None,
) -> T | None:
    return next((x for x in iterable if predicate(x)), default)


def find_or_raise(
    iterable: Iterable[T],
    predicate: Callable[[T], bool],
    exception: Exception | Callable[[], Exception] | None = None,
) -> T:
    found = find(iterable, predicate)
    if found is not None:
        return found

    if exception is None:
        raise ValueError("Element not found")
    if isinstance(exception, Exception):
        raise exception
    raise exception()


def get_tensorboard_logger_or_raise(trainer: Trainer) -> TensorBoardLogger:
    return cast(
        TensorBoardLogger,
        find_or_raise(
            trainer.loggers,
            lambda l: isinstance(l, TensorBoardLogger),
            ValueError("TensorBoard logger not found"),
        ),
    )


def get_tensorboard_logger(
    trainer: Trainer,
    default: TensorBoardLogger | None = None,
) -> TensorBoardLogger | None:
    return cast(
        TensorBoardLogger | None,
        find(
            trainer.loggers,
            lambda l: isinstance(l, TensorBoardLogger),
            default,
        ),
    )
