import torch
import torch.nn as nn
import tqdm
import logging
from typing import List
from collections import deque


def anneal_dsm_score_estimation(scorenet: nn.Module, samples: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """
    Return the loss of NCSN model. Reference: https://github.com/ermongroup/ncsnv2/blob/master/losses/dsm.py, line 3.

    Args:
        scorenet (nn.Module): a score network that takes in a sample and a label and outputs a score.
        samples (torch.Tensor): a tensor of shape (n_samples, *sample_shape)
        sigmas (torch.Tensor): The standard deviations of different noise levels. Shape (num_classes,)
    Returns:
        loss (Tensor): The loss of NCSN model. Shape: torch.Size([])
    """
    # NOTE: 原代码中labels是一个形状为(samples.shape[0],)的随机张量，元素取值为从0到num_classes-1的整数，代表给当前batch中每个样本分配的噪声级别的索引。
    # NOTE: 原代码将labels作为一个可选参数，当前代码为了简洁性去除了该参数，使用原代码中的默认生成方式。
    labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device) # Shape: (n_samples,)
    used_sigmas = sigmas[labels] # Choose noise standard deviations for each sample in the batch. Shape: (n_samples,)
    used_sigmas = used_sigmas.view(samples.shape[0], *([1] * len(samples.shape[1:]))) # -> (n_samples, 1, 1, 1)
    noise = torch.randn_like(samples) * used_sigmas # Shape: (n_samples, n_channels, height, width)
    perturbed_samples = samples + noise
    target = - 1 / (used_sigmas ** 2) * noise
    scores = scorenet(perturbed_samples, labels) # Shape: (n_samples, n_channels, height, width)
    target = target.view(target.shape[0], -1) # -> (n_samples, n_channels * height * width)
    scores = scores.view(scores.shape[0], -1) # -> (n_samples, n_channels * height * width)

    # NOTE: 源代码中将anneal_power作为一个可选参数，但论文中已证明该值为2最优，因此这里直接使用该值，不作为可传入参数。
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** 2 # We use $\lambda(\sigma)$ to balance the loss, and it is proved that $\lambda(\sigma)=\sigma^2$

    return loss.mean(dim=0)


@torch.no_grad()
def anneal_Langevin_dynamics(x_mod: torch.Tensor, scorenet: nn.Module, sigmas: torch.Tensor,
                                n_steps_each: int=200, step_lr: float=0.000008,
                                final_only: bool=False, verbose: bool=False, denoise: bool=True) -> List[torch.Tensor]:
    """
    Return the denoised samples using annealed Langevin dynamics. Reference: https://github.com/ermongroup/ncsnv2/blob/master/models/__init__.py, line 20.

    Args:
        x_mod (Tensor): Input (noisy) data. Shape: (batch_size, *sample_shape).
        scorenet (nn.Module): Score network to compute the gradient.
        sigmas (Tensor): Noise levels. Shape: (num_classes,).
        n_steps_each (int): Number of steps to take for each noise level.
        step_lr (float): Step size for each step.
        final_only (bool): If True, only return the final denoised image.
        verbose (bool): If True, print the step information.
        denoise (bool): If True, denoise the final image using the last noise level.
    
    Returns:
        out (list of Tensors):
            - if final_only is True: Final denoised samples. Length: 1.
            - if final_only is False: Denoised samples at each step. Length:
                - num_classes * n_steps_each + 1 (if denoise=True).
                - num_classes * n_steps_each (if denoise=False).
    """
    images = []

    with torch.no_grad():
        for c, sigma in tqdm.tqdm(enumerate(sigmas), desc='Sampling...'): # Iterate over noise levels
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c # Compute the noise labels for each sample
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each): # Iterate over steps within each noise level
                grad = scorenet(x_mod, labels) # Compute gradient given samples and noise labels. Shape: (n_samples, *sample_shape)
                noise = torch.randn_like(x_mod) # (n_samples, *sample_shape)
                x_mod = x_mod + step_size * grad + noise * torch.sqrt(step_size * 2)

                if not final_only:
                    images.append(x_mod.to('cpu'))
                if verbose:
                    grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                    noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                    image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                    snr = torch.sqrt(step_size / 2.) * grad_norm / noise_norm # Signal to Noise Ratio
                    grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2
                    print("level: {}, step_size: {:.12f}, grad_norm: {:.12f}, image_norm: {:.12f}, snr: {:.12f}, grad_mean_norm: {:.12f}".format(
                        c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

        if denoise: # Denoise the final image using the last noise level
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images


# NOTE: 这是待测方法的实现
@torch.no_grad()
def FC_ALD(x_mod: torch.Tensor, scorenet: nn.Module, sigmas: torch.Tensor,
                                n_steps_each: int=200, step_lr: float=0.000008,
                                k_p: float=1.0, k_i: float=0.0, k_d: float=0.0,
                                final_only: bool=False, verbose: bool=False, denoise: bool=True) -> List[torch.Tensor]:
    """
    Return the denoised samples using `feedback-controlled annealed Langevin dynamics`.

    Args:
        x_mod (Tensor): Input (noisy) data. Shape: (batch_size, *sample_shape).
        scorenet (nn.Module): Score network to compute the gradient.
        sigmas (Tensor): Noise levels. Shape: (num_classes,).
        n_steps_each (int): Number of steps to take for each noise level.
        step_lr (float): Step size for each step.
        k_p (float): Proportional gain for the feedback control.
        k_i (float): Integral gain for the feedback control.
        k_d (float): Derivative gain for the feedback control.
        final_only (bool): If True, only return the final denoised image.
        verbose (bool): If True, print the step information.
        denoise (bool): If True, denoise the final image using the last noise level.
    
    Returns:
        out (list of Tensors):
            - if final_only is True: Final denoised samples. Length: 1.
            - if final_only is False: Denoised samples at each step. Length:
                - num_classes * n_steps_each + 1 (if denoise=True).
                - num_classes * n_steps_each (if denoise=False).
    """
    images = []

    with torch.no_grad():
        for c, sigma in enumerate(sigmas): # Iterate over noise levels
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c # Compute the noise labels for each sample
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2

            # NOTE: 以下为修改处；其它地方与anneal_Langevin_dynamics的函数实现完全相同
            e_int=torch.zeros_like(x_mod).to(x_mod.device)
            e_prev=torch.zeros_like(x_mod).to(x_mod.device)
            e_t=torch.zeros_like(x_mod).to(x_mod.device)
            for t in range(n_steps_each): # Iterate over steps within each noise level
                grad = scorenet(x_mod, labels) # Compute gradient given samples and noise labels. Shape: (n_samples, *sample_shape)
                
                if t>0:
                    e_prev=e_t # update e_prev to e_t of the previous step
                e_int += grad * step_size
                e_t = grad
                
                noise = torch.randn_like(x_mod) # (n_samples, *sample_shape)
                x_mod = x_mod + step_size * (k_p * grad + k_i * e_int + k_d * (e_t - e_prev) / step_size) + noise * torch.sqrt(step_size * 2) # NOTE: 核心：样本更新公式
                # NOTE: 以上为修改处

                if not final_only:
                    images.append(x_mod.to('cpu'))
                if verbose:
                    grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                    noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                    image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                    snr = torch.sqrt(step_size / 2.) * grad_norm / noise_norm # Signal to Noise Ratio
                    grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2
                    print("level: {}, step_size: {:.12f}, grad_norm: {:.12f}, image_norm: {:.12f}, snr: {:.12f}, grad_mean_norm: {:.12f}".format(
                        c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

        if denoise: # Denoise the final image using the last noise level
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images


# NOTE: 这是待测方法的实现
@torch.no_grad()
def FC_ALD_v2(x_mod: torch.Tensor, scorenet: nn.Module, sigmas: torch.Tensor,
                                n_steps_each: int=200, step_lr: float=0.000008,
                                k_p: float=1.0, k_i: float=0.0, k_d: float=0.0,
                                k_i_window_size: int=10,
                                final_only: bool=False, denoise: bool=True, verbose: bool=False,
                                log_freq: int=1, logger=None) -> List[torch.Tensor]:
    """
    Return the denoised samples using `feedback-controlled annealed Langevin dynamics`.

    Args:
        x_mod (Tensor): Input (noisy) data. Shape: (batch_size, *sample_shape).
        scorenet (nn.Module): Score network to compute the gradient.
        sigmas (Tensor): Noise levels. Shape: (num_classes,).
        n_steps_each (int): Number of steps to take for each noise level.
        step_lr (float): Step size for each step.
        k_p (float): Proportional gain for the feedback control.
        k_i (float): Integral gain for the feedback control.
        k_d (float): Derivative gain for the feedback control.
        k_i_window_size (int): Moving window size for the integral gain.
        final_only (bool): If True, only return the final denoised image.
        denoise (bool): If True, denoise the final image using the last noise level.
        verbose (bool): If True, print the step information.
        log_freq (int): Interval of logging.
        logger (logging.Logger): Logger for logging. If None, messages will be printed to the console.
    
    Returns:
        out (2-tuple of (list of Tensors) and dict):
            - out[0]:
                - if final_only is True: Final denoised samples. Length: 1.
                - if final_only is False: Denoised samples at each step. Length:
                    - num_classes * n_steps_each + 1 (if denoise=True).
                    - num_classes * n_steps_each (if denoise=False).
            - out[1]: A dictionary of norms and other metrics in the sampling process.
    """
    assert 0<k_i_window_size<=n_steps_each, "k_i_window_size should be in the range (0, n_steps_each)"
    if logger is None:
        out = print
    elif type(logger)==logging.Logger:
        out = logger.info
    else:
        raise ValueError("logger should be None or a logging.Logger object")

    def get_norm(x):
        """Compute the average Frobenius norm over a batch of tensors."""
        return torch.norm(x.view(x.shape[0], -1), dim=-1).mean().item()

    images = []

    grad_norms, e_int_norms, e_diff_norms = [], [], []
    P_term_norms, I_term_norms, D_term_norms = [], [], []
    snrs = []
    image_norms = []
    PID_term_norms, noise_term_norms, delta_term_norms = [], [], []

    with torch.no_grad():
        for c, sigma in enumerate(sigmas): # Iterate over noise levels
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c # Compute the noise labels for each sample
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2

            # NOTE: 以下为修改处；其它地方与anneal_Langevin_dynamics的函数实现完全相同
            e_int=torch.zeros_like(x_mod).to(x_mod.device)
            e_prev=torch.zeros_like(x_mod).to(x_mod.device)
            e_diff=torch.zeros_like(x_mod).to(x_mod.device)
            e_t=torch.zeros_like(x_mod).to(x_mod.device)
            grads = deque(maxlen=k_i_window_size) # Deque: first in, first out.

            for t in range(n_steps_each): # Iterate over steps within each noise level
                grad = scorenet(x_mod, labels) # Compute gradient given samples and noise labels. Shape: (n_samples, *sample_shape)
                grads.append(grad)
                
                # Use a moving window to sum the historical gradients
                e_int = sum(grads) / k_i_window_size
                
                e_prev=e_t # Update e_prev to e_t of the previous step
                e_t = grad
                e_diff = e_t - e_prev if t>0 else torch.zeros_like(x_mod).to(x_mod.device)
                
                noise = torch.randn_like(x_mod) # (n_samples, *sample_shape)
                x_mod = x_mod + step_size * (k_p * grad + k_i * e_int + k_d * e_diff) + noise * torch.sqrt(step_size * 2) # NOTE: 核心：样本更新公式
                
                if not final_only:
                    images.append(x_mod.to('cpu'))
                
                # Compute the norms and record them
                grad_norm = get_norm(grad)
                noise_norm = get_norm(noise)
                snr = torch.sqrt(step_size / 2.).item() * grad_norm / noise_norm # Signal to Noise Ratio
                image_norm = get_norm(x_mod)
                e_int_norm = get_norm(e_int)
                e_diff_norm = get_norm(e_diff)

                P_term = step_size * k_p * grad
                I_term = step_size * k_i * e_int
                D_term = step_size * k_d * e_diff
                PID_term = P_term + I_term + D_term
                noise_term = noise * torch.sqrt(step_size * 2)
                delta_term = PID_term + noise_term
                
                P_term_norm = get_norm(P_term)
                I_term_norm = get_norm(I_term)
                D_term_norm = get_norm(D_term)
                PID_term_norm = get_norm(PID_term)
                noise_term_norm = get_norm(noise_term)
                delta_term_norm = get_norm(delta_term)

                grad_norms.append(grad_norm)
                e_int_norms.append(e_int_norm)
                e_diff_norms.append(e_diff_norm)
                P_term_norms.append(P_term_norm)
                I_term_norms.append(I_term_norm)
                D_term_norms.append(D_term_norm)
                PID_term_norms.append(PID_term_norm)
                noise_term_norms.append(noise_term_norm)
                delta_term_norms.append(delta_term_norm)
                snrs.append(snr)
                image_norms.append(image_norm)

                if verbose:
                    if t % log_freq == 0:
                        message = "level: {}, step: {}, step_size: {:.12f}".format(c, t, step_size)
                        message += ", grad_norm: {:.12f}, noise_norm: {:.12f}, snr: {:.12f}, image_norm: {:.12f}".format(
                            grad_norm, noise_norm, snr, image_norm)
                        message += ", e_int_norm: {:.12f}, e_diff_norm: {:.12f}".format(
                            e_int_norm, e_diff_norm)
                        message += ", P_norm: {:.12f}, I_norm: {:.12f}, D_norm: {:.12f}, PID_norm: {:.12f}".format(
                            P_term_norm, I_term_norm, D_term_norm, PID_term_norm)
                        message += ", noise_term_norm: {:.12f}, delta_term_norm: {:.12f}".format(
                            noise_term_norm, delta_term_norm)
                        out(message)
                # NOTE: 以上为修改处

        if denoise: # Denoise the final image using the last noise level
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.to('cpu'))

        sampler_record_dict = {
            "grad_norms": grad_norms,
            "e_int_norms": e_int_norms,
            "e_diff_norms": e_diff_norms,
            "P_term_norms": P_term_norms,
            "I_term_norms": I_term_norms,
            "D_term_norms": D_term_norms,
            "PID_term_norms": PID_term_norms,
            "noise_term_norms": noise_term_norms,
            "delta_term_norms": delta_term_norms,
            "snrs": snrs,
            "image_norms": image_norms,
        }

        if final_only:
            return [x_mod.to('cpu')], sampler_record_dict
        else:
            return images, sampler_record_dict
