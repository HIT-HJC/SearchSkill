from __future__ import annotations

import os
from pprint import pprint

import hydra
import ray
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.reward_score.searchskill import compute_score


class RewardManager:
    def __init__(self, tokenizer, num_examine: int = 0) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine

    def __call__(self, data: DataProto):
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        printed = 0
        for i in range(len(data)):
            item = data[i]
            prompt_ids = item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = item.batch["responses"]
            valid_response_length = item.batch["attention_mask"][prompt_length:].sum()
            if int(valid_response_length.item()) <= 0:
                continue
            valid_response_ids = response_ids[:valid_response_length]
            response_text = self.tokenizer.decode(valid_response_ids)
            ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = item.non_tensor_batch.get("data_source", "")
            score = compute_score(response_text, ground_truth, data_source=data_source)
            reward_tensor[i, valid_response_length - 1] = score
            if printed < self.num_examine:
                printed += 1
                print(response_text)
                print(f"[searchskill_reward] {score}")
        return reward_tensor


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    if not ray.is_initialized():
        ray_runtime_env = {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}}
        ray_address = os.environ.get("RAY_ADDRESS", "").strip()
        if ray_address:
            ray.init(address=ray_address, runtime_env=ray_runtime_env)
        else:
            ray_init_kwargs = {
                "include_dashboard": False,
                "num_cpus": int(os.environ.get("RAY_NUM_CPUS", "32")),
                "runtime_env": ray_runtime_env,
            }
            ray_num_gpus = os.environ.get("RAY_NUM_GPUS", "").strip()
            if ray_num_gpus:
                ray_init_kwargs["num_gpus"] = int(ray_num_gpus)
            ray_tmp = os.environ.get("RAY_TMPDIR", "").strip()
            if ray_tmp:
                ray_init_kwargs["_temp_dir"] = ray_tmp
            ray.init(**ray_init_kwargs)
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    import verl.trainer.ppo.ray_trainer as ray_trainer
    from search_r1.llm_agent.searchskill_generation import SearchSkillGenerationManager
    from verl.utils import hf_tokenizer
    from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

    ray_trainer.LLMGenerationManager = SearchSkillGenerationManager

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }
    pool_id = "global_pool"
    resource_pool_spec = {pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
    mapping = {Role.ActorRollout: pool_id, Role.Critic: pool_id, Role.RefPolicy: pool_id}

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=int(os.environ.get("SEARCHSKILL_NUM_EXAMINE_TRAIN", "0")))
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=int(os.environ.get("SEARCHSKILL_NUM_EXAMINE_VAL", "1")))
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping),
        ray_worker_group_cls=RayWorkerGroup,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
