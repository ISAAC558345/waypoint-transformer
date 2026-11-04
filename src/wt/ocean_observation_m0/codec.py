"""Measured packet encodings. Lengths come from ``struct`` output, not a restated size."""

from __future__ import annotations

import struct
import zlib
from typing import Dict

MAGIC = b"HS"
VERSION = 1
KIND_PRODUCT = 1
KIND_COMMAND = 2
KIND_ACK = 3
KIND_TELEMETRY = 4

TASK_VISIT = 1
TASK_BACKHAUL = 2
TASK_RETURN = 3
TASK_WAIT = 4

_PRODUCT = struct.Struct("<BBHI qq ffff BBH")
_COMMAND = struct.Struct("<BBIHBhIB")
_ACK = struct.Struct("<BBIHBB")
_TELEMETRY = struct.Struct("<BBHiffffff")


def frame(payload: bytes) -> bytes:
    if not payload:
        raise ValueError("empty payload")
    body = MAGIC + struct.pack("<H", len(payload)) + payload
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return body + struct.pack("<I", crc)


def parse_frame(blob: bytes) -> Dict[str, object]:
    if len(blob) < 8 or blob[:2] != MAGIC:
        return {"ok": False, "reason": "bad_header"}
    length = struct.unpack_from("<H", blob, 2)[0]
    end = 4 + length
    if end + 4 != len(blob):
        return {"ok": False, "reason": "truncated_or_partial", "payload_bytes": max(0, len(blob) - 8)}
    body, crc_bytes = blob[:-4], blob[-4:]
    expected = zlib.crc32(body) & 0xFFFFFFFF
    found = struct.unpack("<I", crc_bytes)[0]
    if expected != found:
        return {"ok": False, "reason": "checksum_fail"}
    payload = blob[4:end]
    return {"ok": True, "reason": "complete", "payload": payload, "framed_bytes": len(blob)}


def encode_product(
    buoy_id: int,
    seq: int,
    window_start_s: float,
    window_end_s: float,
    mean_x_m: float,
    mean_y_m: float,
    hs_m: float,
    measurement_variance_m2: float,
    quality: int,
    n_samples: int,
) -> bytes:
    payload = _PRODUCT.pack(
        VERSION,
        KIND_PRODUCT,
        buoy_id,
        seq,
        int(round(window_start_s * 1000.0)),
        int(round(window_end_s * 1000.0)),
        float(mean_x_m),
        float(mean_y_m),
        float(hs_m),
        float(measurement_variance_m2),
        quality,
        n_samples,
        0,
    )
    return frame(payload)


def decode_product(blob: bytes) -> Dict[str, object]:
    parsed = parse_frame(blob)
    if not parsed["ok"]:
        return parsed
    payload = parsed["payload"]
    if not isinstance(payload, (bytes, bytearray)) or len(payload) != _PRODUCT.size:
        return {"ok": False, "reason": "unexpected_payload_length"}
    values = _PRODUCT.unpack(payload)
    if values[1] != KIND_PRODUCT:
        return {"ok": False, "reason": "unexpected_kind"}
    return {
        "ok": True,
        "reason": "complete",
        "version": values[0],
        "kind": values[1],
        "buoy_id": values[2],
        "seq": values[3],
        "window_start_s": values[4] / 1000.0,
        "window_end_s": values[5] / 1000.0,
        "mean_x_m": values[6],
        "mean_y_m": values[7],
        "hs_m": values[8],
        "measurement_variance_m2": values[9],
        "quality": values[10],
        "n_samples": values[11],
        "application_bytes": _PRODUCT.size,
        "framed_bytes": len(blob),
        "framed_bits": len(blob) * 8,
    }


def encode_command(cmd_id: int, uav_id: int, task_code: int, buoy_id: int, valid_until_s: int) -> bytes:
    payload = _COMMAND.pack(VERSION, KIND_COMMAND, cmd_id, uav_id, task_code, buoy_id, valid_until_s, 0)
    return frame(payload)


def encode_ack(cmd_id: int, uav_id: int, accepted: int) -> bytes:
    payload = _ACK.pack(VERSION, KIND_ACK, cmd_id, uav_id, accepted, 0)
    return frame(payload)


def encode_telemetry(
    uav_id: int,
    time_s: float,
    x_m: float,
    y_m: float,
    airspeed_m_s: float,
    battery_j: float,
    wind_x_m_s: float,
    wind_y_m_s: float,
) -> bytes:
    payload = _TELEMETRY.pack(
        VERSION,
        KIND_TELEMETRY,
        uav_id,
        int(round(time_s)),
        x_m,
        y_m,
        airspeed_m_s,
        battery_j,
        wind_x_m_s,
        wind_y_m_s,
    )
    return frame(payload)


def packet_sizes() -> Dict[str, dict]:
    product = encode_product(1, 1, -1200.0, 0.0, 1.0, 2.0, 1.2, 0.01, 0, 21)
    command = encode_command(1, 0, TASK_VISIT, 1, 1800)
    ack = encode_ack(1, 0, 1)
    telemetry = encode_telemetry(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rows = {}
    for name, blob, application in (
        ("product", product, _PRODUCT.size),
        ("command", command, _COMMAND.size),
        ("ack", ack, _ACK.size),
        ("telemetry", telemetry, _TELEMETRY.size),
    ):
        rows[name] = {
            "application_bytes": application,
            "magic_and_length_bytes": 4,
            "crc32_bytes": 4,
            "framed_bytes": len(blob),
            "framed_bits": len(blob) * 8,
        }
    return rows
