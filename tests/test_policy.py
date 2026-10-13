"""Check the exact joint policy against exhaustive small-graph oracles."""

import itertools

import pytest
import torch

from ours_fjsp.algorithms.policy import SetMatchingPolicy


def _feasible_partial_matchings(mask):
    """Enumerate assignments directly, without the policy's subset recursion."""
    machines, operations = mask.shape
    for assignment in itertools.product(range(-1, operations), repeat=machines):
        selected = [operation for operation in assignment if operation >= 0]
        if not selected or len(selected) != len(set(selected)):
            continue
        if all(operation < 0 or mask[machine, operation]
               for machine, operation in enumerate(assignment)):
            yield assignment


def test_partial_partition_probabilities_and_gradients_match_enumeration():
    mask = torch.tensor([
        [True, True, False],
        [True, False, True],
        [False, True, True],
    ])
    logits = torch.tensor([
        [0.4, -0.3, 100.0],
        [0.2, 100.0, -0.6],
        [100.0, 0.1, -0.8],
    ], dtype=torch.float64, requires_grad=True)
    assignments = list(_feasible_partial_matchings(mask))
    scores = torch.stack([
        torch.stack([logits[machine, operation]
                     for machine, operation in enumerate(assignment)
                     if operation >= 0]).sum()
        for assignment in assignments
    ])
    policy = SetMatchingPolicy(logits, mask)
    expected_log_partition = torch.logsumexp(scores, dim=0)
    torch.testing.assert_close(policy.log_partition(), expected_log_partition)

    log_probabilities = torch.stack([
        policy.log_prob(assignment) for assignment in assignments
    ])
    torch.testing.assert_close(log_probabilities, scores - expected_log_partition)
    assert log_probabilities.exp().sum().item() == pytest.approx(1.0, abs=1e-12)
    assert {sum(operation >= 0 for operation in assignment)
            for assignment in assignments} == {1, 2, 3}
    assert policy.mode() == assignments[int(scores.detach().argmax())]

    # For a Gibbs distribution, d(log Z)/d(edge score) is its occupancy
    # probability. This independent expectation checks all feasible edges.
    expected_gradient = torch.zeros_like(logits)
    for probability, assignment in zip(scores.detach().softmax(dim=0), assignments):
        for machine, operation in enumerate(assignment):
            if operation >= 0:
                expected_gradient[machine, operation] += probability
    gradient, = torch.autograd.grad(policy.log_partition(), logits)
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, expected_gradient)
    assert torch.count_nonzero(gradient[~mask]).item() == 0

    generator = torch.Generator().manual_seed(17)
    for _ in range(24):
        sample = policy.sample(generator=generator)
        assert sample.operations_by_machine in assignments
        torch.testing.assert_close(
            sample.log_probability, policy.log_prob(sample.operations_by_machine)
        )


def test_no_feasible_edge_has_only_the_empty_matching():
    policy = SetMatchingPolicy(torch.zeros((2, 3)), torch.zeros((2, 3), dtype=torch.bool))
    assert policy.cardinality == 0
    assert policy.mode() == (-1, -1)
    sample = policy.sample(generator=torch.Generator().manual_seed(7))
    assert sample.operations_by_machine == (-1, -1)
    assert sample.log_probability.item() == pytest.approx(0.0)
    assert policy.log_prob((-1, -1)).item() == pytest.approx(0.0)
