import torch
import torch.nn.functional as F

from .base_model import BaseModel


def downsample_3D(t, factor=2):
    B, N, H, W = t.shape
    H_new, W_new = H // factor, W // factor
    return t.view(B, N, H_new, factor, W_new, factor).mean(dim=(3, 5))


def get_phasors(nframes, device):
    # Calculate phasor
    phasors_real = torch.cos(
        2 * torch.pi * torch.arange(nframes, device=device) / nframes
    )
    phasors_real = phasors_real[:, None, None]

    phasors_imag = torch.sin(
        2 * torch.pi * torch.arange(nframes, device=device) / nframes
    )
    phasors_imag = phasors_imag[:, None, None]

    return phasors_real, phasors_imag


def pixelbasedprocess_fast(
    im_stack: torch.Tensor,
    mode: int,
    gamma=2,
    phasors_real=None,
    phasors_imag=None,
    stack_order=0,
    mask_on=None,
    mask_off=None,
):
    # im_stack is a pytorch tensor (B x Grid x H x W)
    """
    Process an image stack based on the selected mode.

    Modes:
    - 4: 'epi' (Epifluorescence) - Average projection across frames
    - 10: 'max' - Maximum projection across frames
    - 2: 'max-min' - Range projection (max - min) across frames
    - 1: 'superconfocal' - Superconfocal projection (max + min - gamma * mean)
    - 3: 'std' (Standard Deviation) - Standard deviation projection across frames
    - 0: 'homodyne' - Homodyne processing using phasors
    """
    assert im_stack.dtype == torch.float32, (
        f"Input image stack is {im_stack.dtype}. Should be {torch.float32}"
    )

    # Normalize each stack
    im_stack = im_stack / im_stack.view(im_stack.size(0), -1).mean(
        dim=1, keepdim=True
    ).view(-1, 1, 1)

    if (
        mode == 0
    ):  # 'homodyne' - Homodyne processing using phasors, this is because scanning can start at position 3 for example
        if phasors_real is not None and phasors_imag is not None:
            # Roll according to order, roll has ti be backwards
            phasors_real = torch.roll(phasors_real, shifts=-stack_order, dims=0)
            phasors_imag = torch.roll(phasors_imag, shifts=-stack_order, dims=0)

            im_processed = torch.sqrt(
                torch.sum(im_stack * phasors_real, dim=0) ** 2
                + torch.sum(im_stack * phasors_imag, dim=0) ** 2
            )

        else:
            print(
                "Error: phasors_real and phasors_imag are required for 'homodyne' mode."
            )
            return None
    elif mode == 1:  # 'superconfocal' - Superconfocal projection
        im_processed = (
            torch.max(im_stack, dim=0).values
            + torch.min(im_stack, dim=0).values
            - gamma * torch.mean(im_stack, dim=0)
        )
    elif mode == 2:  # 'max-min' - Range projection (max - min) across frames
        im_processed = (
            torch.max(im_stack, dim=0).values - torch.min(im_stack, dim=0).values
        )
    elif mode == 3:  # 'epi' - Average projection across frames
        im_processed = torch.mean(im_stack, dim=0)
    elif mode == 4:  # Scaled Substraction
        if mask_on is not None:
            # Compute the numerator and denominator for the beta calculation
            # N = im_stack.size(0)  # Number of frames
            sum_mask_on = torch.sum(mask_on, dim=0)
            sum_mask_off = torch.sum(mask_off, dim=0)

            # beta = (N * torch.sum(mask_on * mask_off, dim=0) - (sum_mask_on * sum_mask_off))/(N * torch.sum(mask_on**2, dim=0) - sum_mask_on**2)
            beta = 1

            # Compute scaled subtraction
            im_processed = beta * (
                torch.sum(im_stack * mask_on, dim=0) / sum_mask_on
                - torch.sum(im_stack * mask_off, dim=0) / sum_mask_off
            )

        else:
            print(
                "Error: Calibration pattern is required for Scaled Substraction algorithm."
            )
            return None

    elif mode == 10:  # 'max' - Maximum projection across frames
        im_processed = torch.max(im_stack, dim=0).values
    elif mode == 11:  # 'std' - Standard deviation projection
        im_processed = torch.std(im_stack, dim=0)
    else:
        print("Invalid mode")
        return None

    return im_processed


def microscope_simulator_fast(
    im_estimate: torch.Tensor,
    PSF_fft: torch.Tensor,
    exc_pat_moves,
    downsample,
    padd: bool = True,
):
    nscansteps, H_hr, W_hr = exc_pat_moves.shape

    # excited sample
    excited_sample = im_estimate * exc_pat_moves

    if padd:
        padH = H_hr // 16
        padW = W_hr // 16

        # Pad the excited sample
        excited_sample = F.pad(
            excited_sample, (padW, padW, padH, padH)
        )  # padd the excited sample for convolution

        H_pad, W_pad = excited_sample.shape[-2:]

        # Pad the PSF fft
        PSF_fft = F.pad(
            PSF_fft,
            (0, W_pad // 2 + 1 - PSF_fft.shape[-1], 0, H_pad - PSF_fft.shape[-2]),
        )

    H_pad, W_pad = excited_sample.shape[-2:]

    # Do the convolution in batch! Much faster
    result = torch.fft.irfft2(
        torch.fft.rfft2(excited_sample) * PSF_fft.unsqueeze(0), s=(H_pad, W_pad)
    )
    result = torch.abs(torch.fft.ifftshift(result, dim=(-2, -1)))

    if padd:
        result = result[:, padH:-padH, padW:-padW]

    # Return the stack, downsampled for later calculation of ratio with original im_stack
    return downsample_3D(result, factor=downsample)


def microscope_simulator_inverse_fast(
    stack_guess: torch.Tensor,
    PSF_fft: torch.Tensor,
    exc_pat_moves_hr,
    upsample,
    padd: bool = True,
):
    B, nscansteps, H_lr, W_lr = stack_guess.shape
    H_hr, W_hr = H_lr * upsample, W_lr * upsample

    # Upsample the input stack (N, H, W)
    stack_guess_hr = F.interpolate(
        stack_guess, scale_factor=upsample, mode="bilinear", align_corners=False
    )

    if padd:
        padH = H_hr // 16
        padW = W_hr // 16

        # First padd the stack_guess
        stack_guess_hr = F.pad(stack_guess_hr, (padW, padW, padH, padH))

        H_pad, W_pad = stack_guess_hr.shape[-2:]

        # Then pad the PSF fft for propper convolution
        PSF_fft = F.pad(
            PSF_fft,
            (0, W_pad // 2 + 1 - PSF_fft.shape[-1], 0, H_pad - PSF_fft.shape[-2]),
        )

    H_pad, W_pad = stack_guess_hr.shape[-2:]

    # Do the convolution in batch! Much faster
    result = torch.fft.irfft2(
        torch.fft.rfft2(stack_guess_hr) * PSF_fft.unsqueeze(0), s=(H_pad, W_pad)
    )
    result = torch.abs(torch.fft.ifftshift(result, dim=(-2, -1)))

    if padd:
        result = result[:, padH:-padH, padW:-padW]

    # Multiply by the excitation pattern
    estimate = result * exc_pat_moves_hr
    return estimate.sum(dim=0) / (H_hr * W_hr)


class JRL(BaseModel):
    def __init__(self, upscale: int, n_iter: int = 10):
        self.upscale = upscale
        self.n_iter = n_iter

    def forward(
        self,
        pixel_values: torch.Tensor,
        calib: torch.Tensor,
        psf: torch.Tensor,
        im_init: torch.Tensor | None = None,
    ):
        assert (
            pixel_values.dtype == torch.float32
            and calib.dtype == torch.float32
            and psf.dtype == torch.float32
        )
        if im_init is not None:
            assert im_init.dtype == torch.float32

        # Dimensions
        B, nscansteps, H_lr, W_lr = pixel_values.shape

        # Initial estimate
        if im_init is not None:
            estimate = F.interpolate(
                im_init, scale_factor=self.upscale, mode="bilinear", align_corners=False
            )
        else:
            pr, pi = get_phasors(nscansteps, pixel_values.device)
            estimate_lr = pixelbasedprocess_fast(
                pixel_values,
                mode=0,
                gamma=2,
                phasors_real=pr,
                phasors_imag=pi,
                stack_order=0,
            )
            estimate_tensor = estimate_lr
            print(f"{estimate_tensor.shape=}")
            estimate = F.interpolate(
                estimate_tensor[None, ...],
                scale_factor=self.upscale,
                mode="bilinear",
                align_corners=False,
            )

        # Upsample excitation patterns and PSF in case both are given at the same ressolution as im_stack
        if calib.shape[1] <= pixel_values.shape[1] and self.upscale > 1:
            calib = F.interpolate(
                calib, scale_factor=self.upscale, mode="bilinear", align_corners=False
            )

        if psf.shape[1] <= pixel_values.shape[1] and self.upscale > 1:
            psf = F.interpolate(
                psf[None, None, ...],
                scale_factor=self.upscale,
                mode="bilinear",
                align_corners=False,
            )

        PSF_fft = torch.fft.rfft2(psf)

        # Precompute normalization term
        H_ones = microscope_simulator_inverse_fast(
            torch.ones_like(pixel_values),
            PSF_fft,
            calib,
            upsample=self.upscale,
            padd=False,
        )

        # Iterative RL updates
        for i in range(self.n_iter):
            print(f"Iteration {i + 1}/{self.n_iter}")
            stack_sim = microscope_simulator_fast(
                estimate, PSF_fft, calib, downsample=self.upscale, padd=False
            )
            stack_sim = stack_sim + 1e-6  # Avoid division by zero

            ratio = pixel_values / stack_sim
            Hr = microscope_simulator_inverse_fast(
                ratio, PSF_fft, calib, upsample=self.upscale, padd=False
            )

            estimate = estimate * (Hr / H_ones)

        # print(f"GPU Time {time.time() - start} s")

        estimate = estimate / torch.max(estimate)
        torch.cuda.synchronize() if estimate.device.type == "cuda" else None
        print("Deconvolution done.")
        return estimate
