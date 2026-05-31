import torch
import torch.nn.functional as F

from .MicroscPSF import microscPSF


def phase_optim_AOD_fast(
    image_in: torch.Tensor,
    laser_profile: torch.Tensor,
    num_iters: int,
    device: torch.device | str,
):
    desired_pattern = torch.sqrt(image_in).to(device=device)
    laser_profile = torch.sqrt(laser_profile).to(device=device)

    phase_ini = torch.hann_window(image_in.shape[1], periodic=False) * 2 * torch.pi
    phase_ini = phase_ini.unsqueeze(0).expand(image_in.shape[0], -1).to(device=device)

    A = torch.fft.ifftshift(
        torch.fft.ifft(
            torch.fft.fftshift(desired_pattern * torch.exp(1j * phase_ini), dim=1),
            dim=1,
        ),
        dim=1,
    )
    for _ in range(num_iters):
        B = torch.abs(laser_profile) * torch.exp(1j * torch.angle(A))
        C = torch.fft.fftshift(torch.fft.fft(B, dim=1), dim=1)
        D = torch.abs(desired_pattern) * torch.exp(1j * torch.angle(C))
        A = torch.fft.ifft(torch.fft.ifftshift(D, dim=1), dim=1)

    return A


def shift_fourier2D(im: torch.Tensor, delta: tuple, pad=512):
    """
    Shift an image or stack of images laterally using a linear phase

    im_in: is a torch tensor
    delta: corresponds to the shift, it can be a float also
    """
    # Ensure batch dimension
    is_batch = im.ndim == 3
    if im.ndim == 2:
        im = im.unsqueeze(0)  # shape -> (1, H, W)
    elif not is_batch:
        raise ValueError("Input tensor must be 2D or 3D")

    N, H, W = im.shape
    dx, dy = delta

    # Apply padding if requested
    if pad > 0:
        im = F.pad(im, (pad, pad, pad, pad))  # (left, right, top, bottom)
        H += 2 * pad
        W += 2 * pad

    # Frequency grids
    uy = torch.fft.fftfreq(H, d=1.0, device=im.device).reshape(H, 1)  # (H, 1)
    vx = torch.fft.fftfreq(W, d=1.0, device=im.device).reshape(1, W)  # (1, W)

    # Linear phase term
    phase = torch.exp(-2j * torch.pi * (dx * vx + dy * uy))  # (H, W)

    # Apply shift in Fourier domain
    result = torch.fft.ifft2(torch.fft.fft2(im) * phase).abs()

    # Crop back if padded
    if pad > 0:
        result = result[..., pad:-pad, pad:-pad]

    # Remove batch if input was 2D
    if not is_batch:
        result = result[0]

    return result


def generate_psf(num_slices, H, W, pix_size_lat, pix_size_axi, wavelength, NA, device):
    H_psf, W_psf = H // 16, W // 16

    psf = microscPSF(
        H_psf,
        W_psf,
        num_slices,
        NA=NA,
        wavelength=wavelength,
        res_lateral=pix_size_lat,
        res_axial=pix_size_axi,
    ).to(dtype=torch.float32, device=device)
    psf = torch.abs(psf) ** 2
    psf /= torch.max(psf)

    # Pad to (Z, H, W)
    pad_H = H - H_psf
    pad_W = W - W_psf
    pad_top = pad_H // 2
    pad_bottom = pad_H - pad_top
    pad_left = pad_W // 2
    pad_right = pad_W - pad_left

    psf = F.pad(psf, (pad_left, pad_right, pad_top, pad_bottom))
    return psf


def foundation_pattern(pattern_type, H_pix, W_pix, incr, n_steps):
    intensity_x = torch.zeros(W_pix)
    intensity_y = torch.zeros(H_pix)

    indx = torch.arange(0, W_pix, incr)
    indy = torch.arange(0, H_pix, incr)

    if pattern_type == "multipoint":
        intensity_x[indx] = 1.0
        intensity_y[indy] = 1.0
        n_steps_x = n_steps
        n_steps_y = n_steps
    elif pattern_type == "multiline":
        intensity_x[:] = 1.0
        intensity_y[indy] = 1.0
        n_steps_x = 1
        n_steps_y = n_steps
    elif pattern_type == "const_square":
        square_size_x = round(0.3 * W_pix / 2)
        centerx = W_pix // 2

        square_size_y = round(0.3 * H_pix / 2)
        centery = H_pix // 2

        idx = torch.arange(-square_size_x // 2, square_size_x // 2 + 1) + centerx
        idx = torch.clip(idx, 0, W_pix - 1)
        idy = torch.arange(-square_size_y // 2, square_size_y // 2 + 1) + centery
        idy = torch.clip(idy, 0, H_pix - 1)
        intensity_x[idx] = 1.0
        intensity_y[idy] = 1.0
        n_steps_x = 1
        n_steps_y = 1
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}")

    return intensity_x.unsqueeze(0), intensity_y.unsqueeze(0), n_steps_x, n_steps_y


def create_holograms(
    pattern_type,
    H_pix,
    W_pix,
    incr,
    n_steps,
    phase_optim,
    n_aods,
    device,
    iterations=300,
):
    intensity_x, intensity_y, n_steps_x, n_steps_y = foundation_pattern(
        pattern_type, H_pix, W_pix, incr, n_steps
    )

    if n_aods == 1:
        if pattern_type == "multipoint":
            raise ValueError("n_aods=1 don't support 'multipoint'")
        holos_x = torch.zeros([1, W_pix], dtype=torch.complex64)
        holos_x[0, W_pix // 2] = 1.0

        # Still calculate holos_y
        if phase_optim == 1:
            laser_int_y = torch.ones_like(intensity_y)
            holos_y = phase_optim_AOD_fast(intensity_y, laser_int_y, iterations, device)
        else:
            holos_y = torch.fft.fftshift(
                torch.fft.ifft(
                    torch.fft.ifftshift(torch.sqrt(intensity_y), dim=1), dim=1
                ),
                dim=1,
            )
    else:
        if phase_optim == 1:
            laser_int_x = gaussian_at_1e2(W_pix)
            laser_int_y = gaussian_at_1e2(H_pix)
            holos_x = phase_optim_AOD_fast(intensity_x, laser_int_x, iterations, device)
            holos_y = phase_optim_AOD_fast(intensity_y, laser_int_y, iterations, device)
        else:
            holos_x = torch.fft.fftshift(
                torch.fft.ifft(
                    torch.fft.ifftshift(torch.sqrt(intensity_x), dim=1), dim=1
                ),
                dim=1,
            )
            holos_y = torch.fft.fftshift(
                torch.fft.ifft(
                    torch.fft.ifftshift(torch.sqrt(intensity_y), dim=1), dim=1
                ),
                dim=1,
            )

    return holos_x, holos_y, intensity_x, intensity_y, n_steps_x, n_steps_y


def gauss_fun1D(A, xx, xc, FWHM):
    sigma = FWHM / (2 * torch.sqrt(2 * torch.log(torch.tensor(2))))
    g = A * torch.exp(-((xx - xc) ** 2) / (2 * sigma**2))
    return g


def gaussian_at_1e2(npix):
    x = torch.arange(npix)
    xc = npix / 2
    FWHM = npix  # 1/e² diameter ≈ FWHM (in pixel units)
    sigma = FWHM / (2 * torch.sqrt(2 * torch.log(torch.tensor(2))))

    return torch.exp(-((x - xc) ** 2) / (2 * sigma**2))


def upsample_3D(t, scale_factor=2, mode="bilinear"):
    return F.interpolate(
        t.unsqueeze(1), scale_factor=scale_factor, mode=mode, align_corners=False
    ).squeeze(1)


def downsample_3D(t, factor=2):
    N, H, W = t.shape
    H_new, W_new = H // factor, W // factor
    return t.view(N, H_new, factor, W_new, factor).mean(dim=(2, 4))


def upsample_2D(t, scale_factor=2, mode="bilinear"):
    return F.interpolate(
        t[None, None, ...], scale_factor=scale_factor, mode=mode, align_corners=False
    )[0, 0]


def downsample_2D(t, factor=2):
    H, W = t.shape
    H_new, W_new = H // factor, W // factor
    return t.view(H_new, factor, W_new, factor).mean(dim=(1, 3))


def adjust_periodicity(H_simu, pix_size_simu, periodicity):
    incr0 = periodicity / pix_size_simu
    divisors = [d for d in range(1, H_simu + 1) if H_simu % d == 0]
    incr = min(divisors, key=lambda d: abs(d - incr0))

    periodicity_new = incr * pix_size_simu

    return periodicity_new, incr
