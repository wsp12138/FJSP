"""Initialize the active networks with the paper experiment's random stream.

The archived experiment constructed several temporary networks and initialized
unused heads before creating its final graph actor. Removing them changes the
initial tensors and the random stream used for subsequent action sampling.
This module reproduces only their ordered Linear initializations, immediately
discarding those temporary tensors. It contains no historical policy, forward
path, training code, checkpoint, or import from the original repository.

Compatibility is defined for the paper's CPU float32 initialization and fixed
6/5/15 feature profile, using the same PyTorch version and CPU random generator.
Networks can subsequently be moved to the requested execution device. This
does not promise identical training trajectories across PyTorch versions or
hardware backends.
"""
from __future__ import annotations

import torch
from torch import nn

from .networks import BipartiteSetActor, ValueCritic


INITIALIZATION_PROFILE = "paper_spec_cpu_float32_reference_v1"


def _discard_linears(*shapes: tuple[int, int, bool]) -> None:
    """Consume the reference initialization draws without retaining modules."""
    for inputs, outputs, bias in shapes:
        nn.Linear(inputs, outputs, bias=bias, device="cpu", dtype=torch.float32)


def _discard_edge_bootstrap(context: int, hidden: int) -> None:
    _discard_linears((context, hidden, True), (hidden, hidden, True),
                     (hidden, 1, False))


def _reset_linears(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            child.reset_parameters()


def _zero_last_linear(module: nn.Sequential) -> None:
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)


def initialize_paper_networks(
    *, obs_dim: int, state_dim: int, action_dim: int, num_agents: int,
    max_jobs: int, hidden_dim: int, value_scale: float,
    global_feature_dim: int, machine_feature_dim: int, op_feature_dim: int,
) -> tuple[BipartiteSetActor, ValueCritic]:
    """Return active actor/critic and preserve the reference post-init RNG state.

    Draws depend on the machine count, state width, job capacity and hidden
    width. No seed is set here: the caller owns torch.manual_seed, exactly as
    in the formal training protocol.
    """
    if torch.get_default_dtype() != torch.float32 or torch.empty(0).device.type != "cpu":
        raise ValueError("Paper reference initialization requires CPU float32 defaults")
    if (global_feature_dim, machine_feature_dim, op_feature_dim) != (6, 5, 15):
        raise ValueError("Paper reference initialization requires the 6/5/15 profile")
    hidden = hidden_dim
    wait_input = global_feature_dim + num_agents + machine_feature_dim
    pair_input = wait_input + op_feature_dim
    # The constructor bootstrap first initialized two three-layer scorers.
    _discard_linears(
        (wait_input, hidden, True), (hidden, hidden, True), (hidden, 1, True),
        (pair_input, hidden, True), (hidden, hidden, True), (hidden, 1, True),
    )
    # The retained value branch precedes the final actor in the reference.
    critic = ValueCritic(state_dim, hidden, value_scale)
    # Three unused critic outputs were initialized after its value output.
    _discard_linears((hidden, num_agents, True), (hidden, max_jobs, True),
                     (hidden, 1, True))
    edge_context = global_feature_dim + machine_feature_dim + op_feature_dim
    _discard_edge_bootstrap(edge_context, hidden)
    _discard_edge_bootstrap(edge_context, hidden)
    token_embedding = max(16, hidden // 2)
    _discard_linears(
        (global_feature_dim + op_feature_dim, token_embedding, True),
        (4, token_embedding, True),
        (token_embedding * 2, hidden, True), (hidden, 1, False),
    )
    # Allocate the clean module without consuming the stream twice. Its active
    # layers are reset below in reference order; no removed module is retained.
    with torch.random.fork_rng(devices=[]):
        actor = BipartiteSetActor(
            obs_dim=obs_dim, action_dim=action_dim, num_agents=num_agents,
            hidden_dim=hidden, global_feature_dim=global_feature_dim,
            machine_feature_dim=machine_feature_dim, op_feature_dim=op_feature_dim,
        )
    for name in ("global_encoder", "machine_encoder", "operation_encoder", "edge_encoder",
                 "machine_update", "operation_update"):
        _reset_linears(getattr(actor, name))
    actor.machine_norm.reset_parameters()
    actor.operation_norm.reset_parameters()
    embedding = actor.embedding_dim
    residual_hidden = max(16, embedding // 2)
    # The disabled context update sat between node updates and edge queries.
    _discard_linears((embedding * 3, embedding, True), (embedding, embedding, True))
    _reset_linears(actor.machine_query)
    _reset_linears(actor.operation_key)
    _reset_linears(actor.edge_bias)
    _reset_linears(actor.load_residual)
    _zero_last_linear(actor.load_residual)
    _reset_linears(actor.route_residual)
    _zero_last_linear(actor.route_residual)
    # An alternative was deep-copied (no draws), followed by two disabled gates.
    route_input = embedding * 4 + 8
    _discard_linears(
        (route_input, residual_hidden, True), (residual_hidden, 1, True),
        (route_input, residual_hidden, True), (residual_hidden, 1, True),
    )
    _reset_linears(actor.balanced_route_residual)
    _zero_last_linear(actor.balanced_route_residual)
    _reset_linears(actor.wait_head)
    return actor, critic
