"""Bit-exact capture codec. E17's public manifest is a summary and is not restored through this path."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import torch


def _pack_array(value: np.ndarray) -> dict:
    array = np.ascontiguousarray(value)
    if array.dtype == object:
        raise TypeError("object arrays are not part of the session capture")
    return {
        "__ndarray__": base64.b64encode(array.tobytes()).decode("ascii"),
        "dtype": str(array.dtype),
        "shape": [int(dim) for dim in array.shape],
    }


def encode(value):
    """Turn a capture into JSON-safe objects. Tuples stay tagged so RNG states restore."""
    if isinstance(value, dict):
        if any(key in value for key in ("__ndarray__", "__tuple__", "__tensor__")):
            raise TypeError("capture collided with a codec tag")
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return {"__tuple__": [encode(item) for item in value]}
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, torch.Tensor):
        packed = _pack_array(value.detach().cpu().contiguous().numpy())
        packed["__tensor__"] = True
        packed["torch_dtype"] = str(value.dtype).replace("torch.", "")
        return packed
    if isinstance(value, np.ndarray):
        return _pack_array(value)
    if isinstance(value, np.generic):
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        raise TypeError(f"unsupported numpy scalar {type(value)!r}")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported capture value {type(value)!r}")


def decode(value):
    if isinstance(value, list):
        return [decode(item) for item in value]
    if isinstance(value, dict):
        if "__ndarray__" in value:
            raw = base64.b64decode(value["__ndarray__"])
            array = np.frombuffer(raw, dtype=np.dtype(value["dtype"])).reshape(tuple(value["shape"])).copy()
            if value.get("__tensor__"):
                tensor = torch.from_numpy(array)
                dtype = getattr(torch, str(value["torch_dtype"]))
                if tensor.dtype != dtype:
                    tensor = tensor.to(dtype=dtype)
                return tensor
            return array
        if "__tuple__" in value:
            return tuple(decode(item) for item in value["__tuple__"])
        return {str(key): decode(item) for key, item in value.items()}
    return value


def dump_capture(path: Path, payload: dict) -> str:
    encoded = encode(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(encoded, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    path.write_text(text + "\n", encoding="utf-8")
    return text


def load_capture(path: Path) -> dict:
    return decode(json.loads(path.read_text(encoding="utf-8")))
