import torch
import numpy as np
from scipy.special import j1
import torch.nn.functional as F


def pixelbasedprocess_fast(im_stack, mode, gamma = 2, phasors_real=None, phasors_imag=None, device = 'cuda', stack_order = 0, mask_on = None, mask_off = None):
    # im_stack is a pytorch tensor (Nframes, Ny, Nx)
    
    '''
    Process an image stack based on the selected mode.
    
    Modes:
    - 4: 'epi' (Epifluorescence) - Average projection across frames
    - 10: 'max' - Maximum projection across frames
    - 2: 'max-min' - Range projection (max - min) across frames
    - 1: 'superconfocal' - Superconfocal projection (max + min - gamma * mean)
    - 3: 'std' (Standard Deviation) - Standard deviation projection across frames
    - 0: 'homodyne' - Homodyne processing using phasors
    '''
    
    target_device = torch.device(device if device == 'cuda' and torch.cuda.is_available() else 'cpu')
    im_stack = torch.from_numpy(im_stack).to(dtype=torch.float32, device=target_device)
    
    # Normalize each stack
    im_stack = (im_stack / im_stack.view(im_stack.size(0), -1).mean(dim=1, keepdim=True).view(-1, 1, 1))


    if mode == 0:  # 'homodyne' - Homodyne processing using phasors, this is because scanning can start at position 3 for example
        if phasors_real is not None and phasors_imag is not None:
            
            # Roll according to order, roll has ti be backwards
            phasors_real = torch.roll(phasors_real, shifts=-stack_order, dims=0)
            phasors_imag = torch.roll(phasors_imag, shifts=-stack_order, dims=0)
            
            im_processed = torch.sqrt(torch.sum(im_stack * phasors_real, dim=0)**2 +
                                      torch.sum(im_stack * phasors_imag, dim=0)**2)
            
        else:
            print("Error: phasors_real and phasors_imag are required for 'homodyne' mode.")
            return None
    elif mode == 1:  # 'superconfocal' - Superconfocal projection
        im_processed = torch.max(im_stack, dim=0).values + torch.min(im_stack, dim=0).values - gamma * torch.mean(im_stack, dim=0)   
    elif mode == 2:  # 'max-min' - Range projection (max - min) across frames
        im_processed = torch.max(im_stack, dim=0).values - torch.min(im_stack, dim=0).values
    elif mode == 3:  # 'epi' - Average projection across frames
        im_processed = torch.mean(im_stack, dim=0)
    elif mode == 4: # Scaled Substraction

        if mask_on is not None:
    
            # Compute the numerator and denominator for the beta calculation
            N = im_stack.size(0)  # Number of frames
            sum_mask_on = torch.sum(mask_on, dim=0)
            sum_mask_off = torch.sum(mask_off, dim=0)
    
            # beta = (N * torch.sum(mask_on * mask_off, dim=0) - (sum_mask_on * sum_mask_off))/(N * torch.sum(mask_on**2, dim=0) - sum_mask_on**2)
            beta = 1
        
            # Compute scaled subtraction
            im_processed = beta * (torch.sum(im_stack * mask_on, dim=0) / sum_mask_on - torch.sum(im_stack * mask_off, dim=0) / sum_mask_off)
            
        else:
            print("Error: Calibration pattern is required for Scaled Substraction algorithm.")
            return None

    elif mode == 10:  # 'max' - Maximum projection across frames
        im_processed = torch.max(im_stack, dim=0).values
    elif mode == 11:  # 'std' - Standard deviation projection
        im_processed = torch.std(im_stack, dim=0)
    else:
        print("Invalid mode")
        return None  

    return im_processed.cpu().numpy()

def get_phasors(nframes, device = 'cuda'):
    # Check device, if GPU is available
    
    target_device = torch.device(device if device == 'cuda' and torch.cuda.is_available() else 'cpu')
        
    # Calculate phasor
    phasors_real = torch.cos(2 * torch.pi * torch.arange(nframes, device=target_device) / nframes)
    phasors_imag = torch.sin(2 * torch.pi * torch.arange(nframes, device=target_device) / nframes)
    
    # Send them to device
    phasors_real = phasors_real[:, None, None]
    phasors_imag = phasors_imag[:, None, None]

    return phasors_real, phasors_imag

def PSF_get(opt_params, psf_type="airy", device="cpu"):
    
    """
    Generate a 3D scalar PSF

    opt_params:
        - 'Npix': lateral size (Nx = Ny)
        - 'Nz': number of axial planes
        - 'pix_size': lateral pixel size (µm)
        - 'pix_size_axi': axial pixel size (µm)
        - 'wavelength': wavelength (µm)
        - 'NA': numerical aperture
        - 'n_index': immersion media refractive index
    """
    
    Npix = opt_params['Npix']
    pix_size = opt_params['pix_size']
    pix_size_axi = opt_params['pix_size_axi']
    wavelength = opt_params['wavelength']
    NA = opt_params['NA']
    Nz = opt_params['Nz']
    n_index = opt_params['n_index']

    # Lateral coordinates (in µm)
    x = np.linspace(-Npix//2, Npix//2 - 1 + Npix % 2, Npix) * pix_size
    X, Y = np.meshgrid(x, x, indexing='ij')
    R = np.sqrt(X**2 + Y**2)

    # Create base 2D PSF
    k = 2 * np.pi / wavelength
    kr = k * NA * R

    if psf_type.lower() == "airy":
        with np.errstate(divide='ignore', invalid='ignore'):
            airy = (2 * j1(kr) / kr) ** 2
            airy[kr == 0] = 1.0
        psf_0 = airy
    elif psf_type.lower() == "gaussian":
        omega_0 = 0.84 * wavelength / NA
        psf_0 = np.exp(-2 * (R**2) / (omega_0**2))
    else:
        raise ValueError("Invalid PSF type. Choose 'airy' or 'gaussian'.")

    psf_0 /= psf_0.sum()
    H0 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf_0)))

    # Frequency grid
    fx = np.fft.fftfreq(Npix, d=pix_size)
    FX, FY = np.meshgrid(fx, fx, indexing='ij')
    f_squared = FX**2 + FY**2

    psf_stack = []
    for z in range(Nz):
        z_pos = (z - Nz // 2) * pix_size_axi
        phase = np.exp(1j * 2 * np.pi * z_pos * np.sqrt((n_index / wavelength)**2 - f_squared))
        H_z = H0 * phase
        psf_z = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(H_z)))
        intensity = np.abs(psf_z) ** 2
        intensity /= intensity.sum()
        psf_stack.append(intensity)

    psf_stack = np.stack(psf_stack, axis=0)  # (Nz, Npix, Npix)
    psf_stack = np.transpose(psf_stack, (1, 2, 0))  # → (Npix, Npix, Nz)
    psf_tensor = torch.tensor(psf_stack, dtype=torch.float16, device=device)

    if Nz == 1:
        return psf_tensor[:, :, 0]  # → (Npix, Npix)
    return psf_tensor  # (Npix, Npix, Nz)

def upsample_3D(t, scale_factor=2, mode='bilinear'):
    return F.interpolate(t.unsqueeze(1), scale_factor=scale_factor, mode=mode, align_corners=False).squeeze(1)

def downsample_3D(t, factor=2):
    N, H, W = t.shape
    H_new, W_new = H // factor, W // factor
    return t.view(N, H_new, factor, W_new, factor).mean(dim=(2, 4))

def upsample_2D(t, scale_factor=2, mode='bilinear'):
    return F.interpolate(t[None, None, ...], scale_factor=scale_factor, mode=mode, align_corners=False)[0, 0]

def downsample_2D(t, factor=2):
    H, W = t.shape
    H_new, W_new = H // factor, W // factor
    return t.view(H_new, factor, W_new, factor).mean(dim=(1, 3))


# SIMPLE FUNCTIONS FOR jRL (assuming upscaling and downscaling of 2)
def microscope_simulator(im_estimate, PSF_fft, exc_pat_moves, downsample, device='cpu', padd = True):
    nscansteps, H_hr, W_hr = exc_pat_moves.shape
    
    # excited sample
    excited_sample = im_estimate * exc_pat_moves
    
    if padd:
        padH = H_hr // 16
        padW = W_hr // 16
        
        # Pad the PSF fft
        PSF_fft = F.pad(PSF_fft, (padW, padW, padH, padH))
        excited_sample = F.pad(excited_sample, (padW, padW, padH, padH)) # padd the excited sample for convolution
    
    # Do the convolution in batch! Much faster
    result = torch.fft.ifft2(torch.fft.fft2(excited_sample) * PSF_fft.unsqueeze(0),)
    result = torch.abs(torch.fft.ifftshift(result, dim=(-2, -1)))
    
    if padd:
        result = result[:, padH:-padH, padW:-padW]

    # Return the stack, downsampled for later calculation of ratio with original im_stack
    return downsample_3D(result, factor=downsample)

def microscope_simulator_inverse(stack_guess, PSF_fft, exc_pat_moves_hr, upsample, device='cpu', padd = True):
    nscansteps, H_lr, W_lr = stack_guess.shape
    H_hr, W_hr = H_lr * upsample, W_lr * upsample
    
    # Upsample the input stack (N, H, W)
    stack_guess_hr = upsample_3D(stack_guess, scale_factor=upsample)
    
    if padd:
        padH = H_hr // 16
        padW = W_hr // 16

        # Then pad everything for propper convolution
        PSF_fft = F.pad(PSF_fft, (padW, padW, padH, padH))
        stack_guess_hr = F.pad(stack_guess_hr, (padW, padW, padH, padH))
    
    # Do the convolution in batch! Much faster
    result = torch.fft.ifft2(torch.fft.fft2(stack_guess_hr) * PSF_fft.unsqueeze(0))
    result = torch.abs(torch.fft.ifftshift(result, dim=(-2, -1)))
    
    if padd:
        result = result[:, padH:-padH, padW:-padW]
    
    # Multiply by the excitation pattern
    estimate = result * exc_pat_moves_hr
    return estimate.sum(dim=0) / (H_hr * W_hr)

def jRL_process(im_stack, exc_pat, PSF_em, upsampling, n_iter=10, im_init=None,
                device='cpu', use_fast = False):
    """ This is a wrapper managing DEVICE and FAST flags.
    """
    if device == 'cuda' and torch.cuda.is_available():
        print("Running jRL processing on GPU...")
    else:
        print("Running jRL processing on CPU. This may take a while...")
        device = 'cpu'  # Force CPU if cuda is not available, even if user set it.

    if use_fast:
        print("Using fast implementation.")
        return jRL_process_fast(im_stack, exc_pat, PSF_em, upsampling, n_iter, im_init, device)
    else:
        print("Using slow implementation.")
        return jRL_process_slow(im_stack, exc_pat, PSF_em, upsampling, n_iter, im_init, device)

def jRL_process_slow(im_stack, exc_pat, PSF_em, upsampling, n_iter=10, im_init=None, device='cpu'):
    # start = time.time()
    # Send inputs to device and cast
    im_stack = torch.from_numpy(im_stack).to(dtype=torch.float32, device=device)
    exc_pat = torch.from_numpy(exc_pat).to(dtype=torch.float32, device=device)
    PSF_em = torch.from_numpy(PSF_em).to(dtype=torch.float32, device=device)

    if im_init is not None:
        im_init = torch.from_numpy(im_init).to(dtype=torch.float16, device=device)

    # Dimensions
    nscansteps, H_lr, W_lr = im_stack.shape
    H_hr, W_hr = H_lr * upsampling, W_lr * upsampling

    # Initial estimate
    if im_init is not None:
        estimate = upsample_2D(im_init, scale_factor=upsampling)
    else:
        pr, pi = get_phasors(nscansteps, device=device)
        estimate_lr = pixelbasedprocess_fast(
            im_stack.cpu().numpy(),
            mode=0, gamma=2,
            phasors_real=pr, phasors_imag=pi,
            device=device, stack_order=0
        )
        estimate_tensor = torch.from_numpy(estimate_lr).to(dtype=torch.float32, device=device)
        estimate = upsample_2D(estimate_tensor, scale_factor=upsampling)

    # Upsample excitation patterns and PSF in case both are given at the same ressolution as im_stack
    if exc_pat.shape[1] <= im_stack.shape[1] and upsampling > 1:
        exc_pat = upsample_3D(exc_pat, scale_factor=upsampling)
        
    if PSF_em.shape[1] <= im_stack.shape[1] and upsampling > 1:
        PSF_em = upsample_2D(PSF_em, scale_factor=upsampling)
        
    PSF_fft = torch.fft.fft2(PSF_em)

    # Precompute normalization term
    H_ones = microscope_simulator_inverse(
        torch.ones_like(im_stack), PSF_fft, exc_pat,
        upsample=upsampling, device=device, padd = False
    )

    # Iterative RL updates
    for i in range(n_iter):
        print(f"Iteration {i + 1}/{n_iter}")
        stack_sim = microscope_simulator(
            estimate, PSF_fft, exc_pat,
            downsample=upsampling, device=device, padd = False
        )
        stack_sim = stack_sim + 1e-6  # Avoid division by zero

        ratio = im_stack / stack_sim
        Hr = microscope_simulator_inverse(
            ratio, PSF_fft, exc_pat,
            upsample=upsampling, device=device, padd = False
        )

        estimate = estimate * (Hr / H_ones)
        
    estimate = estimate/torch.max(estimate)
    
    # print(f"GPU Time {time.time() - start} s")
    torch.cuda.synchronize() if device == 'cuda' else None
    print("Deconvolution done.")

    return estimate.cpu().numpy()


def microscope_simulator_fast(im_estimate, PSF_fft, exc_pat_moves, downsample, device='cpu', padd = True):
    nscansteps, H_hr, W_hr = exc_pat_moves.shape
    
    # excited sample
    excited_sample = im_estimate * exc_pat_moves
    
    if padd:
        padH = H_hr // 16
        padW = W_hr // 16
        
        # Pad the excited sample
        excited_sample = F.pad(excited_sample, (padW, padW, padH, padH)) # padd the excited sample for convolution
        
        H_pad, W_pad = excited_sample.shape[-2:]
        
        # Pad the PSF fft
        PSF_fft = F.pad(PSF_fft, (0, W_pad // 2 + 1 - PSF_fft.shape[-1],
                                      0, H_pad - PSF_fft.shape[-2]))

    H_pad, W_pad = excited_sample.shape[-2:]
            
    # Do the convolution in batch! Much faster
    result = torch.fft.irfft2(torch.fft.rfft2(excited_sample) * PSF_fft.unsqueeze(0), s=(H_pad, W_pad))
    result = torch.abs(torch.fft.ifftshift(result, dim=(-2, -1)))
    
    if padd:
        result = result[:, padH:-padH, padW:-padW]

    # Return the stack, downsampled for later calculation of ratio with original im_stack
    return downsample_3D(result, factor=downsample)

def microscope_simulator_inverse_fast(stack_guess, PSF_fft, exc_pat_moves_hr, upsample, device='cpu', padd = True):
    nscansteps, H_lr, W_lr = stack_guess.shape
    H_hr, W_hr = H_lr * upsample, W_lr * upsample
    
    # Upsample the input stack (N, H, W)
    stack_guess_hr = upsample_3D(stack_guess, scale_factor=upsample)
    
    if padd:
        padH = H_hr // 16
        padW = W_hr // 16
        
        # First padd the stack_guess        
        stack_guess_hr = F.pad(stack_guess_hr, (padW, padW, padH, padH))
        
        H_pad, W_pad = stack_guess_hr.shape[-2:]

        # Then pad the PSF fft for propper convolution     
        PSF_fft = F.pad(PSF_fft, (0, W_pad // 2 + 1 - PSF_fft.shape[-1],
                                      0, H_pad - PSF_fft.shape[-2]))
        
    H_pad, W_pad = stack_guess_hr.shape[-2:]
    
    # Do the convolution in batch! Much faster
    result = torch.fft.irfft2(torch.fft.rfft2(stack_guess_hr) * PSF_fft.unsqueeze(0), s=(H_pad, W_pad))
    result = torch.abs(torch.fft.ifftshift(result, dim=(-2, -1)))
    
    if padd:
        result = result[:, padH:-padH, padW:-padW]
    
    # Multiply by the excitation pattern
    estimate = result * exc_pat_moves_hr
    return estimate.sum(dim=0) / (H_hr * W_hr)

def jRL_process_fast(im_stack, exc_pat, psf, upsampling, n_iter=10, im_init=None, device='cpu'):
    # start = time.time()
    # Send inputs to device and cast
    im_stack = torch.from_numpy(im_stack).to(dtype=torch.float32, device=device)
    exc_pat = torch.from_numpy(exc_pat).to(dtype=torch.float32, device=device)
    psf = torch.from_numpy(psf).to(dtype=torch.float32, device=device)

    if im_init is not None:
        im_init = torch.from_numpy(im_init).to(dtype=torch.float32, device=device)

    # Dimensions
    nscansteps, H_lr, W_lr = im_stack.shape
    H_hr, W_hr = H_lr * upsampling, W_lr * upsampling

    # Initial estimate
    if im_init is not None:
        estimate = upsample_2D(im_init, scale_factor=upsampling)
    else:
        pr, pi = get_phasors(nscansteps, device=device)
        estimate_lr = pixelbasedprocess_fast(
            im_stack.cpu().numpy(),
            mode=0, gamma=2,
            phasors_real=pr, phasors_imag=pi,
            device=device, stack_order=0
        )
        estimate_tensor = torch.from_numpy(estimate_lr).to(dtype=torch.float32, device=device)
        estimate = upsample_2D(estimate_tensor, scale_factor=upsampling)

    # Upsample excitation patterns and PSF in case both are given at the same ressolution as im_stack
    if exc_pat.shape[1] <= im_stack.shape[1] and upsampling > 1:
        exc_pat = upsample_3D(exc_pat, scale_factor=upsampling)
        
    if psf.shape[1] <= im_stack.shape[1] and upsampling > 1:
        psf = upsample_2D(psf, scale_factor=upsampling)
    
    PSF_fft = torch.fft.rfft2(psf)

    # Precompute normalization term
    H_ones = microscope_simulator_inverse_fast(
        torch.ones_like(im_stack), PSF_fft, exc_pat,
        upsample=upsampling, device=device, padd = False
    )

    # Iterative RL updates
    for i in range(n_iter):
        print(f"Iteration {i + 1}/{n_iter}")
        stack_sim = microscope_simulator_fast(
            estimate, PSF_fft, exc_pat,
            downsample=upsampling, device=device, padd = False
        )
        stack_sim = stack_sim + 1e-6  # Avoid division by zero

        ratio = im_stack / stack_sim
        Hr = microscope_simulator_inverse_fast(
            ratio, PSF_fft, exc_pat,
            upsample=upsampling, device=device, padd = False
        )

        estimate = estimate * (Hr / H_ones)
    
    # print(f"GPU Time {time.time() - start} s")
    
    estimate = estimate/torch.max(estimate)
    torch.cuda.synchronize() if device == 'cuda' else None
    print("Deconvolution done.")
    return estimate.cpu().numpy()

# PROCESSING IN BATCHES
def jRL_process_batch(im_stack, exc_pat, PSF_em, upsampling, n_iter=10, im_init=None, device='cpu', n_batch = 2):
    # Process large images in NxN steps
    nscansteps, H_lr, W_lr = im_stack.shape
    H_tile_lr = H_lr // n_batch
    W_tile_lr = W_lr // n_batch
    H_tile_hr = H_tile_lr * upsampling
    W_tile_hr = W_tile_lr * upsampling

    # Prepare output
    full_estimate = torch.zeros((H_lr * upsampling, W_lr * upsampling), device=device)

    # Prepare PSF crop (centered)
    PSF_H, PSF_W = PSF_em.shape
    center_y = PSF_H // 2
    center_x = PSF_W // 2
    half_crop_y = H_tile_lr // 2
    half_crop_x = W_tile_lr // 2
    PSF_crop = PSF_em[
        center_y - half_crop_y: center_y + half_crop_y,
        center_x - half_crop_x: center_x + half_crop_x
    ]

    for i in range(n_batch):
        for j in range(n_batch):
            y0_lr = i * H_tile_lr
            y1_lr = (i + 1) * H_tile_lr
            x0_lr = j * W_tile_lr
            x1_lr = (j + 1) * W_tile_lr

            # Extract tiles
            im_stack_tile = im_stack[:, y0_lr:y1_lr, x0_lr:x1_lr]
            exc_pat_tile = exc_pat[:, y0_lr:y1_lr, x0_lr:x1_lr]
            im_init_tile = (
                im_init[y0_lr:y1_lr, x0_lr:x1_lr] if im_init is not None else None
            )

            print(f"Processing tile ({i}, {j})...")

            # Run jRL on tile
            estimate_tile = jRL_process(
                im_stack_tile,
                exc_pat_tile,
                PSF_crop,
                upsampling=upsampling,
                n_iter=n_iter,
                im_init=im_init_tile,
                device=device,
            )

            # Paste back into final HR image
            y0_hr = y0_lr * upsampling
            y1_hr = y0_hr + H_tile_hr
            x0_hr = x0_lr * upsampling
            x1_hr = x0_hr + W_tile_hr

            full_estimate[y0_hr:y1_hr, x0_hr:x1_hr] = torch.from_numpy(estimate_tile).to(device)

    return full_estimate.cpu().numpy()