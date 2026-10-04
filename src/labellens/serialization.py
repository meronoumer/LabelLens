"""Generic dataclass <-> plain-dict (de)serialization, shared by every model.

Every model in LabelLens must serialize cleanly to JSON so trajectories can
become research data. Rather than hand-write to_dict/from_dict per model,
this module introspects dataclasses once and handles the two types plain
`dataclasses.asdict` gets wrong for our purposes: Enums (serialize to their
`.value`, not the member) and datetimes (serialize to ISO 8601 strings).
"""

from __future__ import annotations

import dataclasses
import datetime
import enum
import typing


def to_dict(obj: typing.Any) -> typing.Any:
    """Recursively convert a (possibly nested) dataclass into plain dict/list/str/etc."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, (list, tuple)):
        return [to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {key: to_dict(value) for key, value in obj.items()}
    return obj


def _unwrap_optional(tp: typing.Any) -> typing.Any:
    """Collapse `Optional[X]` (i.e. `Union[X, None]`) down to `X`."""
    if typing.get_origin(tp) is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def from_dict(cls: type, data: typing.Any) -> typing.Any:
    """Reconstruct an instance of dataclass `cls` from a plain dict produced by `to_dict`."""
    if data is None:
        return None
    if dataclasses.is_dataclass(cls):
        hints = typing.get_type_hints(cls)
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name not in data:
                continue
            kwargs[f.name] = _from_dict_value(_unwrap_optional(hints[f.name]), data[f.name])
        return cls(**kwargs)
    return data


def _from_dict_value(field_type: typing.Any, value: typing.Any) -> typing.Any:
    if value is None:
        return None
    origin = typing.get_origin(field_type)
    if origin is list:
        (item_type,) = typing.get_args(field_type)
        item_type = _unwrap_optional(item_type)
        return [_from_dict_value(item_type, item) for item in value]
    if origin is dict:
        return value
    if dataclasses.is_dataclass(field_type):
        return from_dict(field_type, value)
    if isinstance(field_type, type) and issubclass(field_type, enum.Enum):
        return field_type(value)
    if field_type is datetime.datetime:
        return datetime.datetime.fromisoformat(value)
    if field_type is datetime.date:
        return datetime.date.fromisoformat(value)
    return value


class Serializable:
    """Mixin for dataclasses giving them `.to_dict()` / `.from_dict()`."""

    def to_dict(self) -> dict:
        return to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> typing.Any:
        return from_dict(cls, data)
