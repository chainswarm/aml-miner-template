from typing import List, Dict, Any, Type, TypeVar
from enum import IntEnum
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)


def row_to_dict(row: tuple, column_names: List[str]) -> Dict:
    return dict(zip(column_names, row))

def convert_clickhouse_enum(enum_class: Type[IntEnum], value: Any) -> IntEnum:
    if value is None:
        return None
    
    if isinstance(value, enum_class):
        return value
    
    if isinstance(value, int):
        try:
            return enum_class(value)
        except ValueError:
            pass
    
    if isinstance(value, str) and value.isdigit():
        try:
            return enum_class(int(value))
        except ValueError:
            pass
    
    if isinstance(value, str):
        try:
            return enum_class[value.upper()]
        except KeyError:
            pass
    
    available_values = [f"{e.name}({e.value})" for e in enum_class]
    raise ValueError(
        f"Cannot convert '{value}' (type: {type(value).__name__}) to {enum_class.__name__}. "
        f"Available values: {', '.join(available_values)}"
    )