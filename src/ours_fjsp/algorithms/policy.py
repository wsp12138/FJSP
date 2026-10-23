from __future__ import annotations

"""Exact set-valued policy support for a small machine-to-operation graph.

The neural actor supplies one logit per physically feasible edge.  This module
normalizes those logits over *matchings*, rather than over arbitrary orders of
pair tokens.  The support includes every nonempty feasible partial matching when a feasible edge exists.
"""

from dataclasses import dataclass

import torch


def _population_count(value: int) -> int:
    return bin(value).count("1")


def maximum_cardinality(mask: torch.Tensor) -> int:
    """Return the maximum matching size using operation-wise machine bitsets."""
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("mask must be a boolean [machines, operations] tensor")
    machines, operations = mask.shape
    if machines > 16:
        raise ValueError("exact matching policy supports at most 16 machines")
    # The Python bitset calculation is outside the learned scoring path.  It
    # defines only which joint actions are physically capacity maximal.
    edges = mask.detach().cpu().tolist()
    reachable = {0}
    for operation in range(operations):
        next_reachable = set(reachable)
        for used in reachable:
            for machine in range(machines):
                if edges[machine][operation] and not (used & (1 << machine)):
                    next_reachable.add(used | (1 << machine))
        reachable = next_reachable
    return max(_population_count(state) for state in reachable)


@dataclass(frozen=True)
class MatchingSample:
    """One unordered joint action, indexed by machine; -1 means unassigned."""

    operations_by_machine: tuple[int, ...]
    log_probability: torch.Tensor


class SetMatchingPolicy:
    """Gibbs policy over feasible matchings.

    π(A|s) = exp(sum_(m,o)∈A logit[m,o]) / Z(s).

    The partition function is exact for the small machine set used by this
    project.  It is differentiable, so PPO can optimize the *joint* action
    likelihood without a matching solver, teacher, or action-order correction.
    """

    def __init__(
        self, logits: torch.Tensor, mask: torch.Tensor
    ) -> None:
        if logits.ndim != 2 or mask.shape != logits.shape:
            raise ValueError("logits and mask must have the same [M, O] shape")
        if mask.dtype != torch.bool:
            raise ValueError("mask must be boolean")
        if not torch.isfinite(logits[mask]).all():
            raise ValueError("feasible edge logits must be finite")
        if logits.shape[0] > 16:
            raise ValueError("exact matching policy supports at most 16 machines")
        self.logits = logits
        self.mask = mask
        self.machines, self.operations = logits.shape
        self.states = 1 << self.machines
        self.cardinality = maximum_cardinality(mask)
        self.min_cardinality = 1 if self.cardinality else 0
        self._source_indices: list[torch.Tensor] = []
        self._target_indices: list[torch.Tensor] = []
        for machine in range(self.machines):
            source = [state for state in range(self.states) if not state & (1 << machine)]
            self._source_indices.append(
                torch.tensor(source, dtype=torch.long, device=logits.device)
            )
            self._target_indices.append(
                torch.tensor(
                    [state | (1 << machine) for state in source],
                    dtype=torch.long,
                    device=logits.device,
                )
            )
        self._final_states = torch.tensor(
            [
                state
                for state in range(self.states)
                if self.min_cardinality <= _population_count(state) <= self.cardinality
            ],
            dtype=torch.long,
            device=logits.device,
        )
        self._final_cardinalities = torch.tensor(
            [_population_count(int(state)) for state in self._final_states],
            dtype=torch.long, device=logits.device,
        )
        self._forward_tables: list[torch.Tensor] | None = None

    def _forward(self) -> list[torch.Tensor]:
        if self._forward_tables is not None:
            return self._forward_tables
        # A finite sentinel avoids undefined autograd derivatives at
        # logaddexp(-inf, -inf) for unreachable subset states.
        dp = self.logits.new_full((self.states,), -1e9)
        dp = dp.index_fill(0, torch.tensor([0], device=dp.device), 0.0)
        tables = [dp]
        for operation in range(self.operations):
            next_dp = dp
            for machine in range(self.machines):
                if not bool(self.mask[machine, operation]):
                    continue
                source = self._source_indices[machine]
                target = self._target_indices[machine]
                candidate = dp.index_select(0, source) + self.logits[machine, operation]
                combined = torch.logaddexp(next_dp.index_select(0, target), candidate)
                next_dp = next_dp.index_copy(0, target, combined)
            dp = next_dp
            tables.append(dp)
        self._forward_tables = tables
        return tables

    def log_partition(self) -> torch.Tensor:
        return torch.logsumexp(
            self._forward()[-1].index_select(0, self._final_states), dim=0
        )

    def log_partitions_by_cardinality(self) -> torch.Tensor:
        final = self._forward()[-1].index_select(0, self._final_states)
        return torch.stack([
            torch.logsumexp(final[self._final_cardinalities == k], dim=0)
            for k in range(self.min_cardinality, self.cardinality + 1)
        ])

    def log_prob(self, operations_by_machine: tuple[int, ...]) -> torch.Tensor:
        if len(operations_by_machine) != self.machines:
            raise ValueError("joint action length must equal machine count")
        chosen = [(m, o) for m, o in enumerate(operations_by_machine) if o >= 0]
        if not self.min_cardinality <= len(chosen) <= self.cardinality:
            raise ValueError("joint action has invalid cardinality")
        if len({o for _, o in chosen}) != len(chosen):
            raise ValueError("joint action assigns one operation more than once")
        for machine, operation in chosen:
            if operation >= self.operations or not bool(self.mask[machine, operation]):
                raise ValueError("joint action contains an infeasible edge")
        if not chosen:
            return -self.log_partition()
        score = torch.stack([self.logits[m, o] for m, o in chosen]).sum()
        return score - self.log_partition()

    @torch.no_grad()
    def sample(self, *, generator: torch.Generator | None = None) -> MatchingSample:
        tables = self._forward()
        final_logits = tables[-1].index_select(0, self._final_states)
        picked = torch.multinomial(
            torch.softmax(final_logits, dim=0), 1, generator=generator
        ).item()
        state = int(self._final_states[picked])
        assignment = [-1] * self.machines
        for operation in range(self.operations - 1, -1, -1):
            candidates = [tables[operation][state]]
            origins: list[int | None] = [None]
            for machine in range(self.machines):
                if state & (1 << machine) and bool(self.mask[machine, operation]):
                    previous = state ^ (1 << machine)
                    candidates.append(
                        tables[operation][previous] + self.logits[machine, operation]
                    )
                    origins.append(machine)
            choice = torch.multinomial(
                torch.softmax(torch.stack(candidates), dim=0),
                1,
                generator=generator,
            ).item()
            machine = origins[choice]
            if machine is not None:
                assignment[machine] = operation
                state ^= 1 << machine
        if state != 0:
            raise AssertionError("matching sampler did not return to empty state")
        action = tuple(assignment)
        return MatchingSample(action, self.log_prob(action))

    @torch.no_grad()
    def mode(self) -> tuple[int, ...]:
        """Return the highest-scoring feasible matching with fixed tie order."""
        dp = self.logits.new_full((self.states,), float("-inf"))
        dp[0] = 0.0
        back: list[list[tuple[int, int] | None]] = []
        for operation in range(self.operations):
            next_dp = dp.clone()
            choices: list[tuple[int, int] | None] = [None] * self.states
            for machine in range(self.machines):
                if not bool(self.mask[machine, operation]):
                    continue
                for state in range(self.states):
                    if state & (1 << machine) or not torch.isfinite(dp[state]):
                        continue
                    target = state | (1 << machine)
                    score = dp[state] + self.logits[machine, operation]
                    if score > next_dp[target]:
                        next_dp[target] = score
                        choices[target] = (state, machine)
            back.append(choices)
            dp = next_dp
        final_states = [int(s) for s in self._final_states]
        state = max(
            final_states,
            key=lambda s: (
                float(dp[s]), -s,
            ),
        )
        assignment = [-1] * self.machines
        for operation in range(self.operations - 1, -1, -1):
            choice = back[operation][state]
            if choice is not None:
                previous, machine = choice
                assignment[machine] = operation
                state = previous
        if state != 0:
            raise AssertionError("matching mode backtrace did not return to empty state")
        return tuple(assignment)
