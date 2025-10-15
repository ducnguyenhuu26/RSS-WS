from types import ModuleType
from loguru import logger
import loguru
from contextlib import contextmanager
import sys
from typing import Optional, Union
from typing import Sequence

# From: https://github.com/Delgan/loguru/blob/master/loguru/_defaults.py
LOGURU_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS Z}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# Format that includes extras at the end
LOGURU_FORMAT_WITH_EXTRAS = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS Z}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    "<yellow> | {extra}</yellow>"
)

# Type for the filter dictionary, not importable from loguru
# for some reason.
FilterDict = dict[Optional[str], Union[str, int, bool]]

STDOUT_HANDLER_ID: Optional[int] = None


def configure_logger_with_extras():
    """
    Configure the logger to display extras in a pretty format at the end of messages.

    This removes the default handler and adds a new one that shows bound extra fields.
    Call this early in your application to set up the logging format.
    """
    global STDOUT_HANDLER_ID
    # Remove the default handler
    logger.remove()

    # Add the new handler with extras format
    STDOUT_HANDLER_ID = logger.add(
        sys.stderr, format=LOGURU_FORMAT_WITH_EXTRAS, colorize=True
    )


@contextmanager
def change_log_level(changes: dict[str, Sequence[ModuleType | str]]):
    """
    Temporarily change the log level for a specific module.

    Note: This removes the default handler and then adds it back.
    This might lead to loss of logs, so be careful.
    """
    global STDOUT_HANDLER_ID

    level_filtering: FilterDict = {}
    for level, modules in changes.items():
        for module in modules:
            if isinstance(module, ModuleType):
                module_name = module.__name__
            else:
                module_name = module
            level_filtering[module_name] = level

    try:
        if STDOUT_HANDLER_ID is not None:
            logger.remove(STDOUT_HANDLER_ID)
        else:
            logger.remove(0)

        handler_id = None

        try:
            # Add a handler filtered to the specific level
            handler_id = logger.add(
                sys.stderr,
                filter=level_filtering,
                format=LOGURU_FORMAT_WITH_EXTRAS,
                colorize=True,
            )
        except Exception:
            # This branch doesn't really make sense, because it means we've removed the
            # handler that was there before but failed to add a new one. Not sure what to
            # do here, since in principle we cannot log anything if this happens.
            logger.opt(exception=True).error("Error adding log handler, not filtering.")
            yield
        finally:
            if not handler_id:
                return
            logger.remove(handler_id)
            # Restore the default handler
            STDOUT_HANDLER_ID = logger.add(
                sys.stderr, format=LOGURU_FORMAT_WITH_EXTRAS, colorize=True
            )

        yield
    except Exception:
        logger.opt(exception=True).error("Error removing log handler, not filtering.")
        yield
    finally:
        # Nothing to do, so we pass
        pass
