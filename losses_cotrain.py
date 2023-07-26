import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils
from sde_lib import VESDE, VPSDE

GLOBALADVLOSSCOUNT = 0

def get_cotrain_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
    """Create a loss function for training with arbirary SDEs.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      train: `True` for training loss and `False` for evaluation loss.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
      eps: A `float` number. The smallest time step to sample from.

    Returns:
      A loss function.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    std2rat = lambda x: (1 - x ** 2) * ((-2 * 19.9 * torch.log(1 - x ** 2) + 0.1 * 0.1) ** 0.5) / 2 / x

    def loss_fn(model, batch, t, z, mean, std):
        """Compute the loss function.

        Args:
          model: A score model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
        """
        score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
        perturbed_data = mean + std[:, None, None, None] * z
        score = score_fn(perturbed_data, t)

        if not likelihood_weighting:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        loss = torch.mean(losses)
        return loss

    def adv_loss_fn(model, batch, t, z, mean, std, classifier, clsalignonly=False):
        """Compute the loss function.

        Args:
          model: A score model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
        """
        score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)

        if clsalignonly:
            torch.set_grad_enabled(False)

        perturbed_data = mean + std[:, None, None, None] * z
        score = score_fn(perturbed_data, t)
        purified_data = perturbed_data + score * torch.pow(std[:, None, None, None], 2)
        purified_image = sde.img_reconstruct(purified_data, t)

        if clsalignonly:
            torch.set_grad_enabled(True)

            purified_image = purified_image.detach()
            purified_cls_latent = classifier(purified_image, withres=False, withlatent=True)
            return purified_cls_latent

        purified_cls_result, purified_cls_latent = classifier(purified_image, withlatent=True)

        return purified_cls_result, purified_cls_latent, purified_image, std

    def wrapped_loss_fn(model, batch, classifier, classifieroptimizer, label):
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        mean, std = sde.marginal_prob(batch, t)
        z = torch.randn_like(batch)

        adv_noise = torch.zeros_like(batch)
        adv_noise.requires_grad = True
        cls_result, cls_latent, purified_image, weight = adv_loss_fn(model, batch, t, z + adv_noise, mean, std,
                                                                     classifier)
        purified_image = purified_image.detach()
        loss = torch.mean(torch.nn.functional.cross_entropy(cls_result, label))
        cls_latent = cls_latent.detach()
        loss.backward()

        if batch.device.index == 0:
            global GLOBALADVLOSSCOUNT
            GLOBALADVLOSSCOUNT += 1
            if GLOBALADVLOSSCOUNT % 100 == 0:
                print("loss: ", loss)

        adv_noise = adv_noise.grad.sign()
        purturbed_noise = z * 0.995 + adv_noise * 0.1
        cls_latent_adv = adv_loss_fn(model, batch, t,
                                     purturbed_noise / torch.std(purturbed_noise, dim=[1, 2, 3]).view(-1, 1, 1, 1),
                                     mean, std, classifier, clsalignonly=True)
        cls_result, cls_latent = classifier(purified_image, withres=True, withlatent=True)

        clean_dist = torch.mean(
            torch.pow(torch.nn.functional.softmax(cls_result) - torch.nn.functional.one_hot(label, num_classes=10), 2),
            dim=-1)
        clean_adv_dist = torch.mean(torch.pow(cls_latent - cls_latent_adv, 2), dim=-1)

        loss = torch.mean(
            (clean_dist * (1 - torch.pow(weight, 2)) + clean_adv_dist * torch.pow(weight, 2)) * std2rat(weight))
        loss = torch.mean(loss)
        classifieroptimizer.zero_grad()
        loss.backward()
        classifieroptimizer.step()

        model.zero_grad()
        return loss_fn(model, batch, t, z * 0.995 + adv_noise * 0.1, mean, std)

    return wrapped_loss_fn


def get_cotrain_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
    """Create a one-step training/evaluation function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      optimize_fn: An optimization function.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses according to
        https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

    Returns:
      A one-step function for training or evaluation.
    """
    assert continuous, "Discrete training is not supported for cotraining."
    loss_fn = get_cotrain_loss_fn(sde, train, reduce_mean=reduce_mean,
                                  continuous=True, likelihood_weighting=likelihood_weighting)

    def step_fn(state, batch, classifier, classifieroptimizer, label):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
          state: A dictionary of training information, containing the score model, optimizer,
           EMA status, and number of optimization steps.
          batch: A mini-batch of training/evaluation data.

        Returns:
          loss: The average loss value of this state.
        """
        model = state['model']
        if train:
            with torch.enable_grad():
                model.requires_grad_(True)
                optimizer = state['optimizer']

                loss = loss_fn(model, batch, classifier, classifieroptimizer, label)
                optimizer.zero_grad()
                loss.backward()

                optimize_fn(optimizer, model.parameters(), step=state['step'])
                state['step'] += 1
                state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch)
                ema.restore(model.parameters())

        return loss

    return step_fn
