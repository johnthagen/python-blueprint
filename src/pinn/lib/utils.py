from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger


def get_tensorboard_logger_or_raise(trainer: Trainer) -> TensorBoardLogger:
    for logger in trainer.loggers:
        if isinstance(logger, TensorBoardLogger):
            return logger
    raise ValueError("TensorBoard logger not found")


def get_tensorboard_logger(trainer: Trainer) -> TensorBoardLogger | None:
    for logger in trainer.loggers:
        if isinstance(logger, TensorBoardLogger):
            return logger
    return None
