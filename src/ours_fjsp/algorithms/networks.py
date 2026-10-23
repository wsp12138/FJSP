from __future__ import annotations

import torch
from torch import nn


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)


class BipartiteSetActor(nn.Module):
    """Machine-operation graph actor with bidirectional messages and learned edge/wait scores. Active additive scoring heads are retained from the original method."""

    def __init__(self, *, obs_dim: int, action_dim: int, num_agents: int, hidden_dim: int, global_feature_dim: int, machine_feature_dim: int, op_feature_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.max_operations = action_dim - 1
        self.global_feature_dim = global_feature_dim
        self.machine_feature_dim = machine_feature_dim
        self.op_feature_dim = op_feature_dim
        expected = global_feature_dim + num_agents + num_agents * machine_feature_dim + self.max_operations * op_feature_dim
        if obs_dim != expected:
            raise ValueError(f'Expected obs_dim={expected}, got {obs_dim}')
        embedding_dim = max(32, hidden_dim // 4)
        self.embedding_dim = embedding_dim
        self.global_encoder = nn.Sequential(nn.Linear(global_feature_dim, embedding_dim), nn.Tanh(), nn.Linear(embedding_dim, embedding_dim), nn.Tanh())
        self.machine_encoder = nn.Sequential(nn.Linear(machine_feature_dim, embedding_dim), nn.Tanh(), nn.Linear(embedding_dim, embedding_dim), nn.Tanh())
        self.operation_encoder = nn.Sequential(nn.Linear(op_feature_dim + 3, embedding_dim), nn.Tanh(), nn.Linear(embedding_dim, embedding_dim), nn.Tanh())
        self.edge_encoder = nn.Sequential(nn.Linear(4, embedding_dim), nn.Tanh(), nn.Linear(embedding_dim, embedding_dim), nn.Tanh())
        self.machine_update = nn.Sequential(nn.Linear(embedding_dim * 3, embedding_dim), nn.Tanh(), nn.Linear(embedding_dim, embedding_dim))
        self.operation_update = nn.Sequential(nn.Linear(embedding_dim * 3, embedding_dim), nn.Tanh(), nn.Linear(embedding_dim, embedding_dim))
        self.machine_norm = nn.LayerNorm(embedding_dim)
        self.operation_norm = nn.LayerNorm(embedding_dim)
        self.machine_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.operation_key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.edge_bias = nn.Sequential(nn.Linear(embedding_dim * 2, embedding_dim), nn.Tanh(), nn.Linear(embedding_dim, 1))
        residual_hidden = max(16, embedding_dim // 2)
        self.load_residual = nn.Sequential(nn.Linear(8, residual_hidden), nn.Tanh(), nn.Linear(residual_hidden, 1))
        route_input = embedding_dim * 4 + 8
        self.route_residual = nn.Sequential(nn.Linear(route_input, embedding_dim), nn.Tanh(), nn.Linear(embedding_dim, residual_hidden), nn.Tanh(), nn.Linear(residual_hidden, 1))
        self.balanced_route_residual = nn.Sequential(nn.Linear(route_input, embedding_dim), nn.Tanh(), nn.Linear(embedding_dim, residual_hidden), nn.Tanh(), nn.Linear(residual_hidden, 1))
        for head in (self.load_residual, self.route_residual, self.balanced_route_residual):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.wait_head = nn.Sequential(nn.Linear(embedding_dim * 4, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def _parse_joint_obs_batch(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.ndim != 3 or obs.shape[1] != self.num_agents:
            raise ValueError(f'BipartiteSetActor.forward_joint_batch expects complete decision epochs with shape (batch, {self.num_agents}, obs_dim)')
        machine_start = self.global_feature_dim + self.num_agents
        operation_start = machine_start + self.num_agents * self.machine_feature_dim
        batch_size = obs.shape[0]
        global_features = obs[:, 0, :self.global_feature_dim]
        machine_features = obs[:, 0, machine_start:operation_start].reshape(batch_size, self.num_agents, self.machine_feature_dim)
        all_operation_features = obs[:, :, operation_start:].reshape(batch_size, self.num_agents, self.max_operations, self.op_feature_dim)
        representative = all_operation_features[:, 0].clone()
        eligibility = all_operation_features[:, :, :, 4] > 0.5
        processing = all_operation_features[:, :, :, 5]
        return (global_features, machine_features, representative, eligibility, processing)

    def forward_joint(self, obs: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2 or obs.shape[0] != self.num_agents:
            raise ValueError(f'BipartiteSetActor.forward_joint expects one complete decision epoch with shape ({self.num_agents}, obs_dim)')
        if action_mask.ndim != 2 or action_mask.shape[0] != self.num_agents:
            raise ValueError(f'BipartiteSetActor.forward_joint expects an action mask with shape ({self.num_agents}, action_dim)')
        return self.forward_joint_batch(obs.unsqueeze(0), action_mask.unsqueeze(0)).squeeze(0)

    def forward_joint_batch(self, obs: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """Score a minibatch of complete decision epochs in one graph pass."""
        (global_features, machine_features, operation_features, eligibility, processing) = self._parse_joint_obs_batch(obs)
        if action_mask.ndim != 3 or action_mask.shape[:2] != obs.shape[:2]:
            raise ValueError('BipartiteSetActor.forward_joint_batch expects action masks with shape (batch, num_agents, action_dim)')
        batch_size = obs.shape[0]
        dtype = obs.dtype
        exists = operation_features[:, :, 0] > 0.5
        active = exists & (operation_features[:, :, 3] < 0.5)
        feasible_float = eligibility.to(dtype)
        feasible_count = feasible_float.sum(dim=1)
        safe_count = feasible_count.clamp_min(1.0)
        feasible_processing = processing * feasible_float
        mean_processing = feasible_processing.sum(dim=1) / safe_count
        large = torch.finfo(dtype).max / 16.0
        min_processing = torch.where(eligibility, processing, torch.full_like(processing, large)).min(dim=1).values
        max_processing = torch.where(eligibility, processing, torch.zeros_like(processing)).max(dim=1).values
        min_processing = torch.where(feasible_count > 0, min_processing, torch.zeros_like(min_processing))
        encoder_operation_features = operation_features
        operation_augmented = torch.cat([encoder_operation_features, (feasible_count / max(1, self.num_agents)).unsqueeze(-1), mean_processing.unsqueeze(-1), (max_processing - min_processing).unsqueeze(-1)], dim=-1)
        global_embedding = self.global_encoder(global_features)
        machine_embedding = self.machine_encoder(machine_features)
        operation_embedding = self.operation_encoder(operation_augmented)
        # Paper profile: channel 2 is active repair; channel 3 is cumulative load.
        current_load = machine_features[:, :, 3] if self.machine_feature_dim > 3 else torch.zeros(batch_size, self.num_agents, dtype=dtype, device=obs.device)
        minimum_matrix = min_processing.unsqueeze(1).expand_as(processing)
        post_load = current_load.unsqueeze(2) + processing / max(1, self.num_agents)
        edge_features = torch.stack([feasible_float, processing, processing - minimum_matrix, post_load], dim=-1)
        edge_embedding = self.edge_encoder(edge_features)
        active_edges = eligibility & active.unsqueeze(1)
        active_edge_float = active_edges.to(dtype)
        machine_message = ((operation_embedding.unsqueeze(1) + edge_embedding) * active_edge_float.unsqueeze(-1)).sum(dim=2) / active_edge_float.sum(dim=2, keepdim=True).clamp_min(1.0)
        operation_message = ((machine_embedding.unsqueeze(2) + edge_embedding) * feasible_float.unsqueeze(-1)).sum(dim=1) / feasible_float.sum(dim=1).unsqueeze(-1).clamp_min(1.0)
        global_for_machines = global_embedding.unsqueeze(1).expand(-1, self.num_agents, -1)
        global_for_operations = global_embedding.unsqueeze(1).expand(-1, self.max_operations, -1)
        machine_embedding = self.machine_norm(machine_embedding + self.machine_update(torch.cat([machine_embedding, machine_message, global_for_machines], dim=-1)))
        operation_embedding = self.operation_norm(operation_embedding + self.operation_update(torch.cat([operation_embedding, operation_message, global_for_operations], dim=-1)))
        machine_query = self.machine_query(machine_embedding)
        operation_key = self.operation_key(operation_embedding)
        global_edges = global_embedding.reshape(batch_size, 1, 1, -1).expand(-1, self.num_agents, self.max_operations, -1)
        operation_logits = torch.matmul(machine_query, operation_key.transpose(1, 2)) / float(self.embedding_dim) ** 0.5
        operation_logits = operation_logits + self.edge_bias(torch.cat([edge_embedding, global_edges], dim=-1)).squeeze(-1)
        mean_load = current_load.mean(dim=1)
        load_std = current_load.std(dim=1, unbiased=False)
        load_gap = current_load - mean_load.unsqueeze(1)
        processing_increment = processing / max(1, self.num_agents)
        post_load_gap = post_load - mean_load.reshape(batch_size, 1, 1)
        load_residual_features = torch.stack([current_load.unsqueeze(2).expand_as(processing), mean_load.reshape(batch_size, 1, 1).expand_as(processing), load_gap.unsqueeze(2).expand_as(processing), load_std.reshape(batch_size, 1, 1).expand_as(processing), processing_increment, post_load, post_load_gap, torch.abs(post_load_gap)], dim=-1)
        operation_logits = operation_logits + self.load_residual(load_residual_features).squeeze(-1)
        route_residual_features = torch.cat([machine_embedding.unsqueeze(2).expand(-1, self.num_agents, self.max_operations, -1), operation_embedding.unsqueeze(1).expand(-1, self.num_agents, self.max_operations, -1), edge_embedding, global_edges, load_residual_features], dim=-1)
        core_route_residual_logits = self.route_residual(route_residual_features).squeeze(-1)
        route_residual_logits = core_route_residual_logits
        operation_logits = operation_logits + route_residual_logits
        balanced_route_logits = self.balanced_route_residual(route_residual_features).squeeze(-1)
        eligibility_float = eligibility.to(dtype)
        balanced_route_mean = (balanced_route_logits * eligibility_float).sum(dim=1) / eligibility_float.sum(dim=1).clamp_min(1.0)
        balanced_route_logits = (balanced_route_logits - balanced_route_mean.unsqueeze(1)) * eligibility_float
        operation_logits = operation_logits + balanced_route_logits
        # Eq. (16) pools only operations with a currently feasible physical edge.
        wait_pool = active & action_mask[:, :, 1:].any(dim=1)
        active_operation_float = wait_pool.to(dtype).unsqueeze(-1)
        operation_context = (operation_embedding * active_operation_float).sum(dim=1) / active_operation_float.sum(dim=1).clamp_min(1.0)
        machine_context = machine_embedding.mean(dim=1)
        wait_logits = self.wait_head(torch.cat([machine_embedding, global_for_machines, machine_context.unsqueeze(1).expand(-1, self.num_agents, -1), operation_context.unsqueeze(1).expand(-1, self.num_agents, -1)], dim=-1))
        logits = torch.cat([wait_logits, operation_logits], dim=-1)
        return masked_logits(logits, action_mask)


class ValueCritic(nn.Module):
    """The centralized value branch; unused auxiliary heads are omitted."""

    def __init__(self, state_dim: int, hidden_dim: int, value_scale: float):
        super().__init__()
        self.value_scale=value_scale
        self.encoder=nn.Sequential(nn.Linear(state_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,hidden_dim),nn.Tanh())
        self.value_head=nn.Linear(hidden_dim,1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        value=self.value_head(self.encoder(state)).squeeze(-1)
        return value*self.value_scale if self.value_scale!=1.0 else value
