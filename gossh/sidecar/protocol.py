"""IPC wire format for the MoE sidecar.

Transport: Unix domain socket (SOCK_STREAM).
Framing: 4-byte big-endian length prefix + msgpack body.
Tensors: encoded inline as {"shape": [...], "dtype": "float32", "data": bytes}.

For 24-layer gpt-oss-20b with seq_len=128 and hidden_dim=4096 the per-call
payload is ~48MB.  Unix domain sockets on Linux transfer at ~5-10 GB/s, so
the overhead is ~5-10ms — negligible against model inference time.  A shared-
memory path is the natural next optimization; the protocol is designed so that
``data`` can be replaced with an shm handle without changing the message schema.
"""
from __future__ import annotations

import socket
import struct
from typing import Any

import msgpack
import numpy as np
import torch

_HEADER = struct.Struct(">I")   # 4-byte big-endian uint32 message length


# ── Low-level socket helpers ───────────────────────────────────────────────────

def send_msg(sock: socket.socket, payload: dict[str, Any]) -> None:
    """Send a msgpack-encoded dict with a 4-byte length prefix."""
    data = msgpack.packb(payload, use_bin_type=True)
    sock.sendall(_HEADER.pack(len(data)) + data)


def recv_msg(sock: socket.socket) -> dict[str, Any]:
    """Receive a length-prefixed msgpack dict."""
    (length,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
    return msgpack.unpackb(_recv_exact(sock, length), raw=False)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("Socket closed unexpectedly")
        buf.extend(chunk)
    return bytes(buf)


# ── Tensor serialization ───────────────────────────────────────────────────────

def encode_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    """Encode a tensor as a plain dict embeddable in a msgpack message."""
    arr = tensor.detach().cpu().to(torch.float32).numpy()
    return {
        "shape": list(arr.shape),
        "dtype": "float32",
        "data": arr.tobytes(),
    }


def encode_tensor_int(tensor: torch.Tensor) -> dict[str, Any]:
    """Encode an integer tensor (e.g., expert indices)."""
    arr = tensor.detach().cpu().to(torch.int64).numpy()
    return {
        "shape": list(arr.shape),
        "dtype": "int64",
        "data": arr.tobytes(),
    }


def decode_tensor(d: dict[str, Any]) -> torch.Tensor:
    """Decode a tensor from a wire dict."""
    dtype = np.dtype(d["dtype"])
    arr = np.frombuffer(d["data"], dtype=dtype).reshape(d["shape"])
    return torch.from_numpy(arr.copy())


# ── Request / response builders ────────────────────────────────────────────────

def build_route_request(layer_hidden: dict[int, torch.Tensor]) -> dict[str, Any]:
    """Build a ``route`` request message."""
    return {
        "cmd": "route",
        "layers": {
            str(layer_idx): encode_tensor(hidden)
            for layer_idx, hidden in layer_hidden.items()
        },
    }


def parse_route_response(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse a ``route`` response; raises on error."""
    if msg.get("status") != "ok":
        raise RuntimeError(f"Sidecar error: {msg.get('message', 'unknown')}")
    return msg["decisions"]
