"""MoE sidecar worker process entry point.

This module is intended to be called as the ``target`` of a
``multiprocessing.Process``.  It:

1. Builds a ``RouterSidecarModel`` from the provided weight dict.
2. Binds a Unix domain socket and writes a ready-file to signal the parent.
3. Accepts one persistent connection and serves ``route`` requests until the
   client sends ``quit`` or closes the connection.

The worker is a daemon process — it will be killed automatically when the
parent process exits.
"""
from __future__ import annotations

import os
import socket
from pathlib import Path

import torch

from gossh.sidecar.dequant import RouterSidecarModel
from gossh.sidecar.protocol import (
    decode_tensor,
    encode_tensor,
    encode_tensor_int,
    recv_msg,
    send_msg,
)


def _decision_to_dict(decision) -> dict:
    return {
        "layer_idx": decision.layer_idx,
        "selected_experts": encode_tensor_int(decision.selected_experts),
        "expert_weights": encode_tensor(decision.expert_weights),
        "gate_logits": encode_tensor(decision.gate_logits) if decision.gate_logits is not None else None,
        "token_count": decision.token_count,
    }


def run(
    socket_path: str,
    router_weights: dict[int, torch.Tensor],
    top_k: int,
) -> None:
    """Worker main loop.  Blocks until the client closes the connection."""
    model = RouterSidecarModel(router_weights, top_k)
    ready_path = socket_path + ".ready"

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # SO_REUSEADDR does not apply to Unix sockets; unlink is handled by parent.
    server.bind(socket_path)
    server.listen(1)

    # Signal ready to parent
    Path(ready_path).touch()

    conn, _ = server.accept()
    try:
        while True:
            try:
                msg = recv_msg(conn)
            except (EOFError, ConnectionError, OSError):
                break

            cmd = msg.get("cmd")

            if cmd == "route":
                try:
                    layer_hidden = {
                        int(k): decode_tensor(v)
                        for k, v in msg["layers"].items()
                    }
                    decisions = model.route_all(layer_hidden)
                    send_msg(conn, {
                        "status": "ok",
                        "decisions": [_decision_to_dict(d) for d in decisions],
                    })
                except Exception as exc:
                    send_msg(conn, {"status": "error", "message": str(exc)})

            elif cmd == "ping":
                send_msg(conn, {"status": "ok"})

            elif cmd == "quit":
                send_msg(conn, {"status": "ok"})
                break

            else:
                send_msg(conn, {"status": "error", "message": f"unknown cmd: {cmd}"})

    finally:
        conn.close()
        server.close()
        for path in [socket_path, ready_path]:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
