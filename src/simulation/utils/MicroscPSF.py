import torch
from scipy.special import jv as besselj


def microscPSF(
    nx: int,
    ny: int,
    nz: int,
    num_basis: int = 100,
    num_samp: int = 1000,
    oversampling: int = 2,
    NA: float = 1.4,
    wavelength: float = 610e-9,
    ns: float = 1.33,
    ng0: float = 1.5,
    ng: float = 1.5,
    ni0: float = 1.5,
    ni: float = 1.5,
    ti0: float = 150e-6,
    tg0: float = 170e-6,
    tg: float = 170e-6,
    res_lateral: float = 100e-9,
    res_axial: float = 250e-9,
    pz: float = 0,
):
    xp = (nx - 1) / 2
    yp = (ny - 1) / 2
    max_radius = torch.round(torch.sqrt((nx - xp) ** 2 + (ny - yp) ** 2)) + 1
    R: torch.Tensor = torch.arange(oversampling * max_radius) / oversampling

    Ti = ti0 + res_axial * (torch.arange(nz) - (nz - 1.0) / 2.0)

    a = 0
    b = min(1, ns / NA, ni / NA, ni0 / NA, ng0 / NA, ng / NA)
    L = num_samp
    Rho = torch.linspace(a, b, L).unsqueeze(1)

    NN = num_basis
    k0 = 2 * torch.pi / wavelength
    r = R * res_lateral
    A = k0 * NA * r
    A2 = A**2
    Ab = A * b

    # Adjust coefficients as per original scaling
    k00 = 2 * torch.pi / (545e-9)
    factor1 = k0 / k00
    NA0 = 1.4
    factor2 = NA / NA0
    an = 3 * torch.arange(1, NN + 1) - 2
    an = an * factor1 * factor2

    anRho = Rho * an
    J = torch.from_numpy(besselj(0, anRho.numpy()))
    J0A = torch.from_numpy(besselj(0, Ab.numpy()))
    J1A = A * torch.from_numpy(besselj(1, Ab.numpy()))

    anJ0A = torch.outer(J0A, an)
    anb = an * b
    an2 = an**2
    B1anb = torch.from_numpy(besselj(1, anb.numpy()))
    B0anb = torch.from_numpy(besselj(0, anb.numpy()))

    Ele = anJ0A * B1anb - torch.outer(J1A, B0anb)
    domin = an2.unsqueeze(0) - A2.unsqueeze(1)
    Ele = Ele * b / domin

    C1 = ns * pz
    C2 = ni * (Ti - ti0)
    C3 = ng * (tg - tg0)

    OPDs = C1 * torch.sqrt(1 - (NA * Rho / ns) ** 2)
    OPDi = torch.sqrt(1 - (NA * Rho / ni) ** 2) @ C2.unsqueeze(0)
    OPDg = C3 * torch.sqrt(1 - (NA * Rho / ng) ** 2)

    OPD = OPDs + OPDi + OPDg

    W = k0 * OPD
    Ffun = torch.cos(W) + 1j * torch.sin(W)
    Ci = torch.linalg.lstsq(J.to(dtype=torch.complex64), Ffun)[0]

    ciEle = Ele.to(dtype=torch.complex64) @ Ci
    PSF0 = ciEle.real

    # Interpolation from radial to Cartesian grid
    X, Y = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing="ij")
    rPixel = torch.sqrt((X - xp) ** 2 + (Y - yp) ** 2)
    index = torch.floor(rPixel * oversampling).to(dtype=int)
    index = torch.clip(index, 0, len(R) - 2)
    disR = (rPixel - R[index]) * oversampling
    index1 = index
    index2 = index + 1
    disR1 = 1 - disR

    PSF = torch.zeros((nz, nx, ny), dtype=torch.float32)
    for zi in range(nz):
        h = PSF0[:, zi]
        slice_ = h[index2] * disR + h[index1] * disR1
        PSF[zi, :, :] = slice_
    return PSF
