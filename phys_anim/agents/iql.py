
from typing import List, Tuple, Dict
import torch
from torch import Tensor
from isaac_utils import torch_utils
from lightning.fabric import Fabric
from utils.StateActionLib import StateActionLib, MotionStateAction
from phys_anim.agents.models.actor import PPO_Actor
from hydra.utils import instantiate
from phys_anim.agents.models.common import NormObsBase
from phys_anim.envs.env_utils.general import StepTracker
from phys_anim.agents.models.infomax import JointDiscWithMutualInformationEncMLP
from phys_anim.envs.humanoid.humanoid_utils import build_disc_action_observations, compute_humanoid_observations_max

class IQL:

    def __init__(self, fabric: Fabric, config):
        self.all_config = config
        self.config = config.algo.config
        self.w_last = True
        self.fabric = fabric
        self.device: torch.device = fabric.device
        self.discount = self.config.discount
        self.beta = self.config.beta

        self.num_obs = self.all_config.robot.self_obs_size
        self.num_act = self.all_config.robot.number_of_actions

        self.log_dict = {}

        self.discriminator_obs_size_per_step = (
            self.all_config.env.config.discriminator_obs_size_per_step
        )

        self.current_epoch = 0

        self.expectile = self.config.expectile
        self.alpha = self.config.alpha

        self.latent_reset_steps = StepTracker(
            1,
            min_steps=self.config.infomax_parameters.latent_steps_min,
            max_steps=self.config.infomax_parameters.latent_steps_max,
            device=self.device,
        )

        self.setup_character_props()

        self.key_body_ids = torch.tensor(
        [
            self.all_config.robot.dfs_body_names.index(key_body_name)
            for key_body_name in self.all_config.robot.key_bodies
        ],
        dtype=torch.long,
    )

        self.motion_lib :  StateActionLib = StateActionLib(
            motion_file=config.motion_file,
            dof_body_ids = self.dof_body_ids,
            dof_offsets = self.dof_offsets,
            key_body_ids = self.key_body_ids,
            device=self.device,
        )


    def setup(self):
        actor: PPO_Actor = instantiate(
            self.config.actor, num_in=self.num_obs, num_act=self.num_act
        )

        actor_optimizer = instantiate(
            self.config.actor_optimizer,
            params=list(actor.parameters()),
            _convert_="all",
        )

        self.actor, self.actor_optimizer = self.fabric.setup(actor, actor_optimizer)
        self.actor.mark_forward_method("eval_forward")
        self.actor.mark_forward_method("training_forward")

        critic_s: NormObsBase = instantiate(
            self.config.critic_s, num_in=self.num_obs, num_out=1
        )
        critic_optimizer = instantiate(
            self.config.critic_optimizer,
            params=list(critic_s.parameters()),
        )
        self.critic_s, self.critic_s_optimizer = self.fabric.setup(critic_s, critic_optimizer)

        critic_sa: NormObsBase = instantiate(
            self.config.critic_sa, num_in=self.num_obs, num_out=1
        )
        critic_sa_optimizer = instantiate(
            self.config.critic_optimizer,
            params=list(critic_sa.parameters()),
        )
        self.critic_sa, self.critic_sa_optimizer = self.fabric.setup(critic_sa, critic_sa_optimizer)

        target_critic: NormObsBase = instantiate(
            self.config.critic_sa, num_in=self.num_obs , num_out=1
        )

        self.target_critic = self.fabric.setup(target_critic)

        discriminator: JointDiscWithMutualInformationEncMLP = instantiate(
            self.config.discriminator,
            num_in=self.discriminator_obs_size_per_step
                   * self.all_config.env.config.discriminator_obs_historical_steps,
        )
        discriminator_optimizer = instantiate(
            self.config.discriminator_optimizer,
            params=discriminator.parameters(),
        )

        self.discriminator, self.discriminator_optimizer = self.fabric.setup(
            discriminator, discriminator_optimizer
        )

    def fit(self):
        while self.current_epoch < self.config.max_epochs:
            print(f"Epoch: {self.current_epoch}")
            batch = self.collectExpertData()
            '''
            batch = 
            {
                "latents": (self.config.batch_size,sum(self.config.infomax_parameters.latent_dim))
                "StateAction": List[MotionStateAction]
                "reward": (self.config.batch_size, 1)
                "DiscrimObs" : torch.zeros(self.config.batch_size, self.discriminator_obs_size_per_step, device=self.device),
                "HumanoidObservations" : torch.zeros(self.config.batch_size, self.num_obs, device=self.device)
            }
            '''
            self.log_dict.update({"reward": batch["reward"].mean().detach()})
            demo_actions_tensor = torch.zeros(self.config.batch_size, self.num_act, device=self.device)
            for i in range(self.config.batch_size):
                demo_actions_tensor[i] = batch["StateAction"][i].action

            q_value = self.target_critic({"obs": batch["HumanoidObservations"],"actions": demo_actions_tensor, "latents":batch["latents"]})
            v_value = self.critic_s({"obs": batch["HumanoidObservations"],"latents":batch["latents"]})

            q_v_diff = q_value - v_value

            value_loss = torch.where(q_v_diff > 0, self.expectile, (1-self.expectile)) * (q_v_diff**2)

            self.log_dict.update({"value_loss":value_loss.mean().detach()})

            self.critic_s_optimizer.zero_grad()
            self.fabric.backward(value_loss.mean())
            self.critic_s_optimizer.step()

            next_obs = torch.zeros(self.config.batch_size, self.num_obs, device=self.device)
            next_obs[0:self.config.batch_size-1] = batch["HumanoidObservations"][1:self.config.batch_size]
            next_obs[self.config.batch_size-1] = batch["HumanoidObservations"][0]

            next_latents = torch.zeros(self.config.batch_size, sum(self.config.infomax_parameters.latent_dim), device=self.device)
            next_latents[0:self.config.batch_size-1] = batch["latents"][1:self.config.batch_size]
            next_latents[self.config.batch_size-1] = batch["latents"][0]

            q_loss = (batch["reward"]
                      + self.discount * self.critic_s({"obs": next_obs,"latents":next_latents})
                      - self.critic_sa({"obs": batch["HumanoidObservations"],"actions": demo_actions_tensor, "latents":batch["latents"]}))

            self.log_dict.update({"q_loss": q_loss.mean().detach()})

            self.critic_sa_optimizer.zero_grad()
            self.fabric.backward(q_loss.mean())
            self.critic_sa_optimizer.step()

            for param_cur, param_target in zip(self.critic_sa.parameters(), self.target_critic.parameters()):
                param_target.data = (1-self.alpha) * param_target.data + self.alpha * param_cur.data

            self.actor.training = True
            actor_out = self.actor.training_forward({"obs": batch["HumanoidObservations"],"actions": demo_actions_tensor, "latents":batch["latents"]})

            q_val = self.target_critic({"obs": batch["HumanoidObservations"],"actions": demo_actions_tensor, "latents":batch["latents"]})
            v_val = self.critic_s({"obs": batch["HumanoidObservations"],"latents":batch["latents"]})

            adv = torch.minimum(torch.exp(self.beta * (q_val-v_val)), torch.tensor(100.0, device=self.device))

            actor_adw_loss = (adv * actor_out["neglogp"]).mean()

            actor_div_loss, div_loss_log = self.calculate_extra_actor_loss({"obs": batch["HumanoidObservations"],"actions": demo_actions_tensor,"latents":batch["latents"]})

            actor_loss = actor_adw_loss + actor_div_loss

            self.log_dict.update({"actor_loss": actor_loss.mean().detach()})

            self.actor_optimizer.zero_grad()
            self.fabric.backward(actor_loss.mean())
            self.actor_optimizer.step()

            self.actor.training = False
            actor_eval_out = self.actor.eval_forward({"obs": batch["HumanoidObservations"],"latents":batch["latents"]})

            agent_disc_obs = torch.zeros(self.config.batch_size, self.discriminator_obs_size_per_step, device=self.device)
            for i in range(self.config.batch_size):
                agent_disc_obs[i] = build_disc_action_observations(
                    batch["StateAction"][i].root_pos,
                    batch["StateAction"][i].root_rot,
                    batch["StateAction"][i].root_vel,
                    batch["StateAction"][i].root_ang_vel,
                    batch["StateAction"][i].dof_pos,
                    batch["StateAction"][i].dof_vel,
                    batch["StateAction"][i].key_body_pos,
                    torch.zeros(1, device=self.device),
                    actor_eval_out["actions"][i].unsqueeze(0),
                    self.all_config.env.config.humanoid_obs.local_root_obs,
                    self.all_config.env.config.humanoid_obs.root_height_obs,
                    self.all_config.robot.dof_obs_size,
                    self.dof_offsets,
                    False,
                    self.w_last,
                )

            disc_loss , disc_log_dict = self.encoder_step({"AgentDiscObs": agent_disc_obs, "DemoDiscObs":batch["DiscrimObs"], "latents":batch["latents"]})
            self.log_dict.update(disc_log_dict)

            self.discriminator_optimizer.zero_grad()
            self.fabric.backward(disc_loss)
            self.discriminator_optimizer.step()

            self.fabric.log_dict(self.log_dict,self.current_epoch)

            self.current_epoch+=1

            pass


    def sample_latent(self, n):
        latents = torch.zeros(
            [n, sum(self.config.infomax_parameters.latent_dim)], device=self.device
        )

        start = 0
        for idx, dim in enumerate(self.config.infomax_parameters.latent_dim):
            if self.config.infomax_parameters.latent_types[idx] == "gaussian":
                gaussian_sample = torch.normal(latents[:, start : start + dim])
                latents[:, start : start + dim] = gaussian_sample

            elif self.config.infomax_parameters.latent_types[idx] == "hypersphere":
                gaussian_sample = torch.normal(latents[:, start : start + dim])
                projected_gaussian_sample = torch.nn.functional.normalize(
                    gaussian_sample, dim=-1
                )
                latents[:, start : start + dim] = projected_gaussian_sample

            elif self.config.infomax_parameters.latent_types[idx] == "uniform":
                uniform_sample = torch.rand([n, dim], device=self.device)
                latents[:, start : start + dim] = uniform_sample

            elif self.config.infomax_parameters.latent_types[idx] == "categorical":
                categorical_sample = torch.multinomial(
                    latents[0, start : start + dim] + 1.0,
                    num_samples=n,
                    replacement=True,
                )
                b = torch.arange(n, device=self.device)
                latents[b, categorical_sample + start] = categorical_sample
            else:
                raise NotImplementedError

            start += dim

        return latents

    # aka calculate diversity_loss
    def calculate_extra_actor_loss(self, batch_dict) -> Tuple[Tensor, Dict]:
        extra_loss, extra_actor_log_dict = torch.tensor(0.0, device=self.device), {}

        if self.config.infomax_parameters.diversity_bonus <= 0:
            return extra_loss, extra_actor_log_dict

        diversity_loss = self.diversity_loss(batch_dict)

        extra_actor_log_dict["actor/diversity_loss"] = diversity_loss.detach()

        return (
            extra_loss
            + diversity_loss * self.config.infomax_parameters.diversity_bonus,
            extra_actor_log_dict,
        )

    def diversity_loss(self, batch_dict):
        prev_latents = batch_dict["latents"]
        new_latents = (self.sample_latent(batch_dict["obs"].shape[0]))
        batch_dict["latents"] = new_latents
        new_outs = self.actor.training_forward(batch_dict)

        batch_dict["latents"] = prev_latents
        old_outs = self.actor.training_forward(batch_dict)

        clipped_new_mu = torch.clamp(new_outs["mus"], -1.0, 1.0)
        clipped_old_mu = torch.clamp(old_outs["mus"], -1.0, 1.0)

        mu_diff = clipped_new_mu - clipped_old_mu
        mu_diff = torch.mean(torch.square(mu_diff), dim=-1)

        z_diff = new_latents * prev_latents
        z_diff = torch.sum(z_diff, dim=-1)
        z_diff = 0.5 - 0.5 * z_diff

        diversity_bonus = mu_diff / (z_diff + 1e-5)
        diversity_loss = torch.square(
            self.config.infomax_parameters.diversity_tar - diversity_bonus
        ).mean()

        return diversity_loss

    def mi_enc_forward(self, obs: Tensor) -> Tensor:
        args = {"obs": obs}
        return self.discriminator(args, return_enc=True)

    def calc_mi_reward(self, discriminator_obs, latents):
        """
        TODO: calculate reward for each distribution type
            Gaussian -- MSE (we assume variance 1)
            Hypersphere -- von Mises-Fisher
            Uniform -- MSE
            Categorical -- torch.nn.functional.cross_entropy()
        """
        mi_r = torch.zeros(self.config.batch_size, device=self.device)

        enc_pred = self.mi_enc_forward(discriminator_obs)
        cumulative_enc_dim = 0
        cumulative_latent_dim = 0
        for idx, latent_dim in enumerate(self.config.infomax_parameters.latent_dim):
            if self.config.infomax_parameters.latent_types[idx] == "hypersphere":
                r = self.von_mises_fisher_reward(
                    enc_pred[..., cumulative_enc_dim : cumulative_enc_dim + latent_dim],
                    latents[
                        ..., cumulative_latent_dim : cumulative_latent_dim + latent_dim
                    ],
                )

                cumulative_latent_dim += latent_dim
                cumulative_enc_dim += latent_dim
            else:
                raise NotImplementedError

            r = r.squeeze(1)

            mi_r += r * self.config.infomax_parameters.mi_reward_w[idx]

        return mi_r / len(self.config.infomax_parameters.latent_dim)

    def von_mises_fisher_reward(self, enc_prediction, latents):
        neg_err = -self.calc_von_mises_fisher_enc_error(enc_prediction, latents)
        if self.config.infomax_parameters.mi_hypersphere_reward_shift:
            mi_r = (neg_err + 1) / 2
        else:
            mi_r = torch.clamp_min(neg_err, 0.0)
        return mi_r

    def calc_von_mises_fisher_enc_error(self, enc_pred, latent):
        err = enc_pred * latent
        err = -torch.sum(err, dim=-1, keepdim=True)
        return err

    def setup_character_props(self):
        self.dof_body_ids = self.all_config.robot.dfs_dof_body_ids
        self.dof_offsets = []
        previous_dof_name = "null"
        for dof_offset, dof_name in enumerate(self.all_config.robot.dfs_dof_names):
            if dof_name[:-2] != previous_dof_name:  # remove the "_x/y/z"
                previous_dof_name = dof_name[:-2]
                self.dof_offsets.append(dof_offset)
        self.dof_offsets.append(len(self.all_config.robot.dfs_dof_names))
        self.dof_obs_size = self.all_config.robot.dof_obs_size
        self.num_act = self.all_config.robot.number_of_actions

    def build_body_ids_tensor(self, body_names):
        body_ids = []

        for body_name in body_names:
            body_id = self.body_names.index(body_name)
            assert (
                body_id != -1
            ), f"Body part {body_name} not found in {self.body_names}"
            body_ids.append(body_id)

        body_ids = torch_utils.to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def discriminator_forward(self, obs: Tensor, return_norm_obs=False) -> Tensor:
        args = {"obs": obs}
        return self.discriminator(args, return_norm_obs=return_norm_obs)

    def calculate_discriminator_reward(self, discriminator_obs: Tensor) -> Tensor:
        disc_logits = self.discriminator_forward(discriminator_obs)

        prob = 1 / (1 + torch.exp(-disc_logits))
        disc_r = (
            -torch.log(
                torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device))
            )
            * self.config.discriminator_reward_w
        )
        return disc_r

    def collectExpertData(self):
        batch_dict =  {
            "latents" : torch.zeros((self.config.batch_size,sum(self.config.infomax_parameters.latent_dim)),device=self.device),
            "StateAction" : [], #List[MotionStateAction]
            "DiscrimObs" : torch.zeros(self.config.batch_size, self.discriminator_obs_size_per_step, device=self.device),
            "HumanoidObservations" : torch.zeros(self.config.batch_size, self.num_obs, device=self.device)
        }

        latents = self.sample_latent(self.config.batch_size)
        batch_dict["latents"] = latents

        motion_id = self.motion_lib.sample_motions(1)
        motion = self.motion_lib.get_motion(motion_id)
        motion_time = torch.zeros(1,device = self.device)
        for i in range(self.config.batch_size):
            if motion_time < self.motion_lib.get_motion_length(motion_id):
                motion_time += motion.time_delta
                batch_dict["StateAction"].append(self.motion_lib.get_motion_state(motion_id, motion_time))
            else:
                motion_id = self.motion_lib.sample_motions(1)
                motion = self.motion_lib.get_motion(motion_id)
                motion_time = torch.zeros(1,device = self.device)
                batch_dict["StateAction"].append(self.motion_lib.get_motion_state(motion_id, motion_time))

        for i in range(self.config.batch_size):
            batch_dict["DiscrimObs"][i] = build_disc_action_observations(
                batch_dict["StateAction"][i].root_pos,
                batch_dict["StateAction"][i].root_rot,
                batch_dict["StateAction"][i].root_vel,
                batch_dict["StateAction"][i].root_ang_vel,
                batch_dict["StateAction"][i].dof_pos,
                batch_dict["StateAction"][i].dof_vel,
                batch_dict["StateAction"][i].key_body_pos,
                torch.zeros(1,device = self.device),
                batch_dict["StateAction"][i].action,
                self.all_config.env.config.humanoid_obs.local_root_obs,
                self.all_config.env.config.humanoid_obs.root_height_obs,
                self.all_config.robot.dof_obs_size,
                self.dof_offsets,
                False,
                self.w_last,
            )

        for i in range(self.config.batch_size):
            batch_dict["HumanoidObservations"][i] = compute_humanoid_observations_max(
                batch_dict["StateAction"][i].rb_pos,
                batch_dict["StateAction"][i].rb_rot,
                batch_dict["StateAction"][i].rb_vel,
                batch_dict["StateAction"][i].rb_ang_vel,
                torch.zeros(1, device=self.device),
                self.all_config.env.config.humanoid_obs.local_root_obs,
                self.all_config.env.config.humanoid_obs.root_height_obs,
                self.w_last,
            )

        desc_r = self.calculate_discriminator_reward(batch_dict["DiscrimObs"])
        mi_r = self.calc_mi_reward(batch_dict["DiscrimObs"], batch_dict["latents"])

        batch_dict["reward"] = desc_r + mi_r + 1

        return batch_dict

    # batch:{
    #   "AgentDiscObs"
    #   "DemoDiscObs"
    # }
    def discriminator_step(self, batch):
        discriminator_loss, discriminator_log_dict = self.compute_discriminator_loss(
            batch
        )

        discriminator_log_dict = {
            f"jd/{k}": v for k, v in discriminator_log_dict.items()
        }

        return discriminator_loss, discriminator_log_dict

    def compute_discriminator_loss(self, batch):
        (
            agent_obs,
            demo_obs,
        ) = batch["AgentDiscObs"], batch["DemoDiscObs"]

        demo_obs.requires_grad_(True)

        agent_logits = self.discriminator_forward(obs=agent_obs)

        demo_dict = self.discriminator_forward(obs=demo_obs, return_norm_obs=True)
        demo_logits = demo_dict["outs"]
        demo_norm_obs = demo_dict["norm_obs"]

        pos_loss = self.disc_loss_pos(demo_logits)
        agent_loss = self.disc_loss_neg(agent_logits)

        neg_loss = agent_loss

        class_loss = 0.5 * (pos_loss + neg_loss)

        pos_acc = self.compute_pos_acc(demo_logits)
        agent_acc = self.compute_neg_acc(agent_logits)

        neg_acc = agent_acc

        # grad penalty
        disc_demo_grad = torch.autograd.grad(
            demo_logits,
            demo_norm_obs,
            grad_outputs=torch.ones_like(demo_logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        disc_demo_grad_norm = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad_norm)
        grad_loss: Tensor = self.config.discriminator_grad_penalty * disc_grad_penalty

        if self.config.discriminator_weight_decay > 0:
            all_weight_params = self.discriminator.all_discriminator_weights()
            total: Tensor = sum([p.pow(2).sum() for p in all_weight_params])
            weight_decay_loss: Tensor = total * self.config.discriminator_weight_decay
        else:
            weight_decay_loss = torch.tensor(0.0, device=self.device)
            total = torch.tensor(0.0, device=self.device)

        if self.config.discriminator_logit_weight_decay > 0:
            logit_params = self.discriminator.logit_weights()
            logit_total = sum([p.pow(2).sum() for p in logit_params])

            logit_weight_decay_loss: Tensor = (
                logit_total * self.config.discriminator_logit_weight_decay
            )
        else:
            logit_weight_decay_loss = torch.tensor(0.0, device=self.device)
            logit_total = torch.tensor(0.0, device=self.device)

        loss = grad_loss + class_loss + weight_decay_loss + logit_weight_decay_loss

        log_dict = {
            "loss": loss.detach(),
            "pos_acc": pos_acc.detach(),
            "agent_acc": agent_acc.detach(),
            "neg_acc": neg_acc.detach(),
            "grad_penalty": disc_grad_penalty.detach(),
            "grad_loss": grad_loss.detach(),
            "class_loss": class_loss.detach(),
            "l2_logit_total": logit_total.detach(),
            "l2_logit_loss": logit_weight_decay_loss.detach(),
            "l2_total": total.detach(),
            "l2_loss": weight_decay_loss.detach(),
            "demo_logit_mean": demo_logits.detach().mean(),
            "agent_logit_mean": agent_logits.detach().mean(),
        }

        log_dict["negative_logit_mean"] = log_dict["agent_logit_mean"]

        return loss, log_dict

    # batch:{
    #   "AgentDiscObs"
    #   "DemoDiscObs"
    #   "latents"
    # }
    def encoder_step(self,batch) -> Tuple[Tensor, Dict]:
        discriminator_loss, discriminator_log_dict = self.discriminator_step(batch)

        obs = batch["AgentDiscObs"]
        latents = batch["latents"]
        if self.config.infomax_parameters.mi_enc_grad_penalty > 0:
            obs.requires_grad_(True)

        mi_enc_pred = self.mi_enc_forward(obs)

        mi_enc_err = self.calc_mi_enc_error(mi_enc_pred, latents)

        mi_enc_loss = torch.mean(mi_enc_err)

        if self.config.infomax_parameters.mi_enc_weight_decay > 0:
            enc_weight_params = self.discriminator.enc_weights()
            total: Tensor = sum([p.pow(2).sum() for p in enc_weight_params])
            weight_decay_loss: Tensor = (
                total * self.config.infomax_parameters.mi_enc_weight_decay
            )
        else:
            weight_decay_loss = torch.tensor(0.0, device=self.device)

        if self.config.infomax_parameters.mi_enc_grad_penalty > 0:
            mi_enc_obs_grad = torch.autograd.grad(
                mi_enc_err,
                obs,
                grad_outputs=torch.ones_like(mi_enc_err),
                create_graph=True,
                retain_graph=True,
            )

            mi_enc_obs_grad = mi_enc_obs_grad[0]
            mi_enc_obs_grad = torch.sum(torch.square(mi_enc_obs_grad), dim=-1)
            mi_enc_grad_penalty = torch.mean(mi_enc_obs_grad)

            grad_loss: Tensor = mi_enc_grad_penalty * self.config.infomax_parameters.mi_enc_grad_penalty
        else:
            grad_loss = torch.tensor(0.0, device=self.device)

        mi_loss = mi_enc_loss + weight_decay_loss + grad_loss

        log_dict = {
            "loss": mi_loss.detach(),
            "l2_loss": weight_decay_loss.detach(),
            "grad_penalty": grad_loss.detach(),
        }

        mi_enc_log_dict = {f"mi_enc/{k}": v for k, v in log_dict.items()}

        discriminator_log_dict.update(mi_enc_log_dict)

        return mi_loss + discriminator_loss, discriminator_log_dict

    def calc_mi_enc_error(self, enc_pred, latents):
        cumulative_enc_dim = 0
        cumulative_latent_dim = 0

        total_error = []

        for idx, latent_dim in enumerate(self.config.infomax_parameters.latent_dim):
            if self.config.infomax_parameters.latent_types[idx] == "hypersphere":
                err = self.calc_von_mises_fisher_enc_error(enc_pred, latents)

                cumulative_latent_dim += latent_dim
                cumulative_enc_dim += latent_dim
            else:
                raise NotImplementedError

            total_error.append(err)

        return torch.cat(total_error, dim=-1)

    @staticmethod
    def disc_loss_neg(disc_logits) -> Tensor:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            disc_logits, torch.zeros_like(disc_logits)
        )
        return loss

    @staticmethod
    def disc_loss_pos(disc_logits) -> Tensor:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            disc_logits, torch.ones_like(disc_logits)
        )
        return loss

    @staticmethod
    def compute_pos_acc(positive_logit: Tensor) -> Tensor:
        return (positive_logit > 0).float().mean()

    @staticmethod
    def compute_neg_acc(negative_logit: Tensor) -> Tensor:
        return (negative_logit < 0).float().mean()