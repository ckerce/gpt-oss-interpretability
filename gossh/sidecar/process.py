"""MoeSidecar — client-side orchestrator for the MoE router sidecar process.

Usage::

    weights = RouterWeightExtractor().extract(model, structure)
    with MoeSidecar(weights, top_k=arch.top_k) as sidecar:
        capture = InputCapture()
        capture.register(model, structure.mlp_names)
        with torch.no_grad():
            model(input_ids)
        decisions = sidecar.route(capture.captured)
        capture.clear()

Or equivalently via ``from_model``::

    with MoeSidecar.from_model(model, structure, arch_spec) as sidecar:
        ...
"""
from __future__ import annotations

import multiprocessing
import os
import socket
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

from gossh.capture.router_capture import RouterDecision
from gossh.sidecar.dequant import RouterWeightExtractor
from gossh.sidecar.protocol import (
    build_route_request,
    decode_tensor,
    parse_route_response,
    recv_msg,
    send_msg,
)
from gossh.sidecar.worker import run as _worker_run

if TYPE_CHECKING:
    from gossh.backends.structure import ModelStructure
    from gossh.model_registry import ModelArchSpec


def _decision_from_dict(d: dict) -> RouterDecision:
    return RouterDecision(
        layer_idx=d["layer_idx"],
        selected_experts=decode_tensor(d["selected_experts"]),
        expert_weights=decode_tensor(d["expert_weights"]),
        gate_logits=decode_tensor(d["gate_logits"]) if d.get("gate_logits") else None,
        token_count=d["token_count"],
    )


class MoeSidecar:
    """Client-side handle for the MoE router sidecar subprocess.

    The sidecar holds bf16 router weight clones and runs routing decisions
    for hidden states captured from the MXFP4-quantized main forward pass.

    Args:
        router_weights: ``{layer_idx: weight_tensor}`` from
            ``RouterWeightExtractor.extract()``.
        top_k: Number of experts selected per token (from arch spec).
        socket_dir: Directory for the Unix socket file (default: system tmpdir).
        start_timeout: Seconds to wait for the worker to become ready.
    """

    def __init__(
        self,
        router_weights: dict[int, torch.Tensor],
        top_k: int,
        socket_dir: str | None = None,
        start_timeout: float = 15.0,
    ):
        self._router_weights = router_weights
        self._top_k = top_k
        self._socket_dir = socket_dir or tempfile.gettempdir()
        self._socket_path = os.path.join(
            self._socket_dir, f"gossh_sidecar_{uuid.uuid4().hex[:12]}.sock"
        )
        self._ready_path = self._socket_path + ".ready"
        self._start_timeout = start_timeout
        self._process: Optional[multiprocessing.Process] = None
        self._conn: Optional[socket.socket] = None

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        structure: "ModelStructure",
        arch_spec: "ModelArchSpec",
        **kwargs,
    ) -> "MoeSidecar":
        """Construct a MoeSidecar by extracting router weights from a loaded model."""
        weights = RouterWeightExtractor().extract(model, structure)
        if not weights:
            raise RuntimeError(
                "No router weights found. Check that structure.gate_names is populated "
                "and that gate modules exist in model.named_parameters()."
            )
        return cls(weights, top_k=arch_spec.top_k, **kwargs)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn the worker subprocess and wait for it to signal ready."""
        # Clean up any stale socket files from a previous run
        for path in [self._socket_path, self._ready_path]:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

        # Use 'spawn' (fresh interpreter) rather than 'fork' to avoid
        # inheriting pytest's epoll file descriptors, which causes hangs
        # on WSL2 when the parent's socket monitoring is corrupted.
        ctx = multiprocessing.get_context("spawn")
        self._process = ctx.Process(
            target=_worker_run,
            args=(self._socket_path, self._router_weights, self._top_k),
            daemon=True,
        )
        self._process.start()

        # Wait for ready file written by worker after bind+listen
        deadline = time.monotonic() + self._start_timeout
        while time.monotonic() < deadline:
            if Path(self._ready_path).exists():
                break
            if not self._process.is_alive():
                raise RuntimeError("Sidecar worker exited before becoming ready")
            time.sleep(0.02)
        else:
            self._process.terminate()
            raise TimeoutError(
                f"Sidecar worker did not start within {self._start_timeout}s"
            )

        # Connect to worker
        self._conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._conn.connect(self._socket_path)

    def stop(self) -> None:
        """Send quit, join worker, clean up."""
        if self._conn is not None:
            try:
                send_msg(self._conn, {"cmd": "quit"})
                recv_msg(self._conn)
            except Exception:
                pass
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

        if self._process is not None:
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)
            self._process = None

        for path in [self._socket_path, self._ready_path]:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def __enter__(self) -> "MoeSidecar":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    # ── Routing ───────────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Return True if the worker is responsive."""
        if self._conn is None:
            return False
        try:
            send_msg(self._conn, {"cmd": "ping"})
            resp = recv_msg(self._conn)
            return resp.get("status") == "ok"
        except Exception:
            return False

    def route(self, layer_hidden: dict[int, torch.Tensor]) -> list[RouterDecision]:
        """Route a batch of hidden states.

        Args:
            layer_hidden: ``{layer_idx: Tensor[seq_len, hidden_dim]}`` mapping
                from ``InputCapture.captured``.

        Returns:
            List of ``RouterDecision`` objects, one per layer, in layer order.
        """
        if self._conn is None:
            raise RuntimeError("MoeSidecar not started — use as context manager or call start()")

        send_msg(self._conn, build_route_request(layer_hidden))
        resp = recv_msg(self._conn)
        raw_decisions = parse_route_response(resp)
        return [_decision_from_dict(d) for d in raw_decisions]

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def num_layers(self) -> int:
        return len(self._router_weights)

    @property
    def layer_indices(self) -> list[int]:
        return sorted(self._router_weights.keys())

    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()
