from datetime import datetime
from decimal import Decimal
from enum import Enum
from json import JSONEncoder
from typing import Any


class CustomJSONEncoder(JSONEncoder):
    """Custom JSON encoder that handles enums, datetimes, and decimals.

    Extends the standard JSONEncoder to serialize:
    - Enum values to their underlying value
    - datetime objects to ISO format strings
    - Decimal objects to string representation (preserves precision)
    """

    def default(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format.

        Args:
            obj: Object to serialize

        Returns:
            JSON-serializable representation of the object

        Raises:
            TypeError: If object cannot be serialized
        """
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)
