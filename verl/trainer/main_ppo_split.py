# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Split-placement entry point for PPO/GRPO training.

Motivation
----------
When NVIDIA GPU Compute Mode is set to Exclusive_Process, only one OS process
may hold a CUDA context on each GPU at a time.  verl's default colocated layout
puts ActorRollout workers and RefPolicy workers in *separate* Ray actor process
groups but maps both to the same "global_pool" of GPUs — this triggers the
Exclusive_Process restriction.

Solution
--------
This entry point splits the available GPUs into two resource pools:

  Single-node  (nnodes == 1):
    actor_rollout_pool  →  GPU 0 .. N//2-1   (actor + vLLM rollout)
    ref_pool            →  GPU N//2 .. N-1   (ref policy)

  Multi-node  (nnodes >= 2):
    actor_rollout_pool  →  first nnodes//2 nodes (all GPUs on each)
    ref_pool            →  last  nnodes//2 nodes

  If a critic is also needed (PPO, GAE …):
    critic_pool shares the same aux_spec as ref_pool unless you want to add a
    separate critic resource pool.

Usage
-----
    python3 -m verl.trainer.main_ppo_split  <same hydra overrides as main_ppo>

The only additional requirement is that
    actor_rollout_ref.rollout.tensor_model_parallel_size
must equal the number of GPUs in actor_rollout_pool (i.e. N//2 for a single
node), NOT the total GPU count.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.dataset.sampler import AbstractSampler
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler, run_ppo
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


class SplitPlacementTaskRunner(TaskRunner):
    """TaskRunner that places actor/rollout and ref policy on separate GPU pools.

    This avoids CUDA Exclusive_Process mode conflicts that arise when multiple
    Ray process groups compete for the same physical GPUs.

    Pool names
    ----------
    ACTOR_ROLLOUT_POOL : actor + vLLM rollout workers
    REF_POOL           : ref policy workers  (populated only when a ref is needed)
    CRITIC_POOL        : critic workers      (populated only when a critic is needed)
    """

    ACTOR_ROLLOUT_POOL = "actor_rollout_pool"
    REF_POOL = "ref_pool"
    CRITIC_POOL = "critic_pool"

    # ------------------------------------------------------------------
    # Worker registration overrides
    # ------------------------------------------------------------------

    def add_actor_rollout_worker(self, config):
        """Register actor+rollout worker, mapped to ACTOR_ROLLOUT_POOL."""
        from verl.single_controller.ray import RayWorkerGroup

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        if use_legacy_worker_impl == "disable":
            from verl.workers.engine_workers import ActorRolloutRefWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

            lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
            if lora_rank <= 0:
                lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
            ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
            role = Role.ActorRollout if (not need_reference_policy(config) or ref_in_actor) else Role.ActorRolloutRef
            self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
            self.mapping[role] = self.ACTOR_ROLLOUT_POOL
            return actor_rollout_cls, ray_worker_group_cls

        # Legacy worker path (auto / enable)
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError(
                f"Unknown actor strategy: {config.actor_rollout_ref.actor.strategy}"
            )

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        self.mapping[Role.ActorRollout] = self.ACTOR_ROLLOUT_POOL
        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Register critic worker, mapped to CRITIC_POOL (only when critic is needed)."""
        if not need_critic(config):
            return

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        if config.critic.strategy in {"fsdp", "fsdp2"}:
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            else:
                from verl.workers.engine_workers import TrainingWorker as CriticWorker
        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker
        elif config.critic.strategy == "veomni":
            if use_legacy_worker_impl == "disable":
                from verl.workers.engine_workers import TrainingWorker as CriticWorker
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl for veomni: {use_legacy_worker_impl}")
        else:
            raise NotImplementedError(f"Unknown critic strategy: {config.critic.strategy}")

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
        self.mapping[Role.Critic] = self.CRITIC_POOL

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Register ref policy worker on REF_POOL (separate GPUs from actor/rollout)."""
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if use_legacy_worker_impl == "disable":
            # In the new engine, ref is fused into ActorRolloutRefWorker.
            return

        if need_reference_policy(config):
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            # KEY: map to REF_POOL instead of global_pool so ref processes
            # run on physically different GPUs from actor/rollout processes.
            self.mapping[Role.RefPolicy] = self.REF_POOL

    # ------------------------------------------------------------------
    # Resource pool initialisation override
    # ------------------------------------------------------------------

    def init_resource_pool_mgr(self, config):
        """Create split resource pools based on GPU/node counts.

        Single-node split (nnodes == 1):
            actor_rollout_pool: [n_gpus//2]   (first half of GPUs)
            ref/critic pool:    [n_gpus//2]   (second half of GPUs)

        Multi-node split (nnodes >= 2):
            actor_rollout_pool: [n_gpus] * (nnodes//2)
            ref/critic pool:    [n_gpus] * (nnodes//2)
        """
        nnodes = config.trainer.nnodes
        n_gpus = config.trainer.n_gpus_per_node

        if nnodes // 2 == 0 and n_gpus // 2 > 0:
            # Single-node: split GPUs within the node
            actor_spec = [n_gpus // 2] * nnodes
            aux_spec = [n_gpus // 2] * nnodes
        else:
            # Multi-node: split at node boundary
            actor_spec = [n_gpus] * (nnodes // 2)
            aux_spec = [n_gpus] * (nnodes // 2)

        resource_pool_spec = {
            self.ACTOR_ROLLOUT_POOL: actor_spec,
        }

        # Add ref pool only when a ref policy worker was actually registered
        if self.REF_POOL in self.mapping.values():
            resource_pool_spec[self.REF_POOL] = aux_spec

        # Add critic pool only when a critic worker was actually registered
        if self.CRITIC_POOL in self.mapping.values():
            resource_pool_spec[self.CRITIC_POOL] = aux_spec

        # Reward model resource pool (unchanged from base class logic)
        if config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward.reward_model.n_gpus_per_node must be > 0")
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.nnodes must be > 0")
            reward_pool = [config.reward.reward_model.n_gpus_per_node] * config.reward.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
        else:
            # Set reward model to use the actor pool size for compatibility
            config.reward.reward_model.nnodes = nnodes
            config.reward.reward_model.n_gpus_per_node = actor_spec[0]

        print(f"[SplitPlacement] resource_pool_spec: {resource_pool_spec}")
        print(f"[SplitPlacement] role -> pool mapping: {self.mapping}")

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
        )
        return resource_pool_manager


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry for split-placement PPO/GRPO training."""
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    task_runner_class = ray.remote(num_cpus=1)(SplitPlacementTaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
