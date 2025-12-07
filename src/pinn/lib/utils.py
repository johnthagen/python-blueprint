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
    """
    Find the first element in an iterable that satisfies a predicate.

    Args:
        iterable: The iterable to search.
        predicate: A function that returns True for the desired element.
        default: The value to return if no element is found. Defaults to None.

    Returns:
        The first matching element, or the default value.
    """
    return next((x for x in iterable if predicate(x)), default)


def find_or_raise(
    iterable: Iterable[T],
    predicate: Callable[[T], bool],
    exception: Exception | Callable[[], Exception] | None = None,
) -> T:
    """
    Find the first element in an iterable that satisfies a predicate, or raise an exception.

    Args:
        iterable: The iterable to search.
        predicate: A function that returns True for the desired element.
        exception: The exception to raise if no element is found.
                   Can be an Exception instance, a callable returning an Exception,
                   or None (raises ValueError).

    Returns:
        The first matching element.

    Raises:
        ValueError: If no element is found and no specific exception is provided.
        Exception: The provided exception if no element is found.
    """
    found = find(iterable, predicate)
    if found is not None:
        return found

    if exception is None:
        raise ValueError("Element not found")
    if isinstance(exception, Exception):
        raise exception
    raise exception()


def get_tensorboard_logger_or_raise(trainer: Trainer) -> TensorBoardLogger:
    """
    Retrieve the TensorBoardLogger from the trainer, or raise if not present.

    Args:
        trainer: The PyTorch Lightning Trainer instance.

    Returns:
        The TensorBoardLogger.

    Raises:
        ValueError: If no TensorBoardLogger is attached to the trainer.
    """
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
    """
    Retrieve the TensorBoardLogger from the trainer.

    Args:
        trainer: The PyTorch Lightning Trainer instance.
        default: Default value if not found.

    Returns:
        The TensorBoardLogger or the default value.
    """
    return cast(
        TensorBoardLogger | None,
        find(
            trainer.loggers,
            lambda l: isinstance(l, TensorBoardLogger),
            default,
        ),
    )
