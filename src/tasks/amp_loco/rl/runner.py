import os

import torch
import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from rsl_rl.runners.amp_on_policy_runner import AmpOnPolicyRunner


class _OnnxPolicyWrapper(torch.nn.Module):
  """Thin wrapper that exposes ``act_inference`` as ``forward`` for ONNX export.
  
  Includes the obs normalizer so the exported ONNX model expects raw observations
  and C++ deployment does not need to implement normalization separately.
  """

  def __init__(self, actor_critic, obs_normalizer=None):
    super().__init__()
    self.actor_critic = actor_critic
    self.obs_normalizer = obs_normalizer

  def forward(self, obs):
    if self.obs_normalizer is not None:
      obs = self.obs_normalizer(obs)
    return self.actor_critic.act_inference(obs)


class AMPOnPolicyRunner(AmpOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def _export_policy_to_onnx(self, path: str, filename: str = "policy.onnx"):
    """Export the actor network to ONNX using the local ActorCritic model.
    
    The exported model includes the obs normalizer (if empirical_normalization
    is enabled) so that the ONNX model expects raw observations directly.
    """
    policy = self.alg.policy
    # Include normalizer in the ONNX model if empirical normalization is used
    obs_normalizer = None
    if self.empirical_normalization:
      obs_normalizer = self.obs_normalizer
      obs_normalizer.to("cpu")
      obs_normalizer.eval()
    wrapper = _OnnxPolicyWrapper(policy, obs_normalizer)
    wrapper.to("cpu")
    wrapper.eval()
    num_obs = policy.actor[0].in_features
    dummy_input = torch.zeros(1, num_obs)
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      wrapper,
      dummy_input,
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      input_names=["obs"],
      output_names=["actions"],
      dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
    )
    # move policy back to training device
    policy.to(self.device)
    if obs_normalizer is not None:
      obs_normalizer.to(self.device)

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self._export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
