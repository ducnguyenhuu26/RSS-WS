from pathlib import Path
from .log_utils import configure_logger_with_extras

# Configure logging with extras support by default
configure_logger_with_extras()

REPO_ROOT = Path(__file__).parent.parent
