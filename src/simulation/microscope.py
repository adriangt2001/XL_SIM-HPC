import deepinv as dinv
import numpy as np
import torch
import torch.nn.functional as F
import inspect
from .utils import calc_utils


class Microscope:
    ##### Physic Constants (Units) #####

    mm = 1e3
    um = 1
    nm = 1e-3

    ##### Input Params (Input)

    def __init__(
        self,
        pattern_case="multipoint",
        device="cuda",
        device_id=0,
        resolution=(512, 512),
        # cam_height = 512,
        # cam_width = 512,
        cam_pix=6.5,
        binn_simu=2,
        single_plane=True,
        axial_fov=30,
        periodicity=1,
        wavelength_ex=488,
        wavelength_em=520,
        phase_optim=1,
        aod_propag=1,
        numerical_aperture=1.2,
        magnification=60,
        focal=200,
        pix_size_simu_axi=200,
        upsampling_factor=1.05,
        inpainting_mask=0.8,
        noise_level_gaussian=0.1,
        noise_level_poisson=0.1,
    ):
        self.pattern_case = pattern_case
        self.device = torch.device(device, device_id)
        self.cam_height = resolution[0]
        self.cam_width = resolution[1]
        self.cam_pix = cam_pix
        self.binn_simu = binn_simu
        self.single_plane = single_plane
        self.fov_axi = axial_fov
        self.periodicity_raw = periodicity
        self.wavelength_ex = wavelength_ex
        self.wavelength_em = wavelength_em
        self.phase_optim = phase_optim
        self.aod_propag = aod_propag
        self.na = numerical_aperture
        self.magn = magnification
        self.f_obj_raw = focal
        self.pix_size_simu_axi = pix_size_simu_axi
        self.upsampling_factor = upsampling_factor
        self.inpainting_mask = inpainting_mask
        self.noise_level_gaussian = noise_level_gaussian
        self.noise_level_poisson = noise_level_poisson

        self._periodicity, self._increment = None, None
        self.psf_ex, self.psf_em = None, None
        self._calib_pattern, self._num_steps = None, None

        ##### Expensive Variables (Main Variables)
        self._periodicity, self._increment = self.compute_periodicity()
        self.psf_ex, self.psf_em = self.generate_psf()
        self._calib_pattern, self._num_steps = self.generate_pattern()

        print("Microscope ready!")

    ##### Physic Values (Input with units) #####

    @property
    def _cam_pix(self):
        return self.cam_pix * self.um

    @property
    def _fov_axi(self):
        return self.fov_axi * self.um

    @property
    def _periodicity_raw(self):
        return self.periodicity_raw * self.um

    @property
    def _wavelength_ex(self):
        return self.wavelength_ex * self.nm

    @property
    def _wavelength_em(self):
        return self.wavelength_em * self.nm

    @property
    def _f_obj_raw(self):
        return self.f_obj_raw * self.mm

    @property
    def _pix_size_simu_axi(self):
        return self.pix_size_simu_axi * self.nm

    ##### Computed Params (Properties) #####

    @property
    def _cam_size_simu(self):
        return torch.tensor(
            [self.cam_height * self.binn_simu, self.cam_width * self.binn_simu]
        )

    @property
    def _num_aods(self):
        return 2 if self.pattern_case == "multipoint" else 1

    @property
    def _f_obj(self):
        return self._f_obj_raw / self.magn

    @property
    def _pix_size(self):
        return self._cam_pix / self.magn

    @property
    def _pix_size_simu(self):
        return self._pix_size / self.binn_simu

    @property
    def _num_slices(self):
        if not self.single_plane:
            num_slices = int(self._fov_axi / self._pix_size_simu_axi)
            if num_slices % 2 == 0:
                num_slices += 1
        else:
            num_slices = 1
        return num_slices

    @property
    def _airy_unit_ex(self):
        return 1.22 * self._wavelength_ex / self.na

    @property
    def _airy_unit_em(self):
        return 1.22 * self._wavelength_em / self.na

    @property
    def _fwhm_ex(self):
        return 0.51 * self._wavelength_ex / self.na

    @property
    def _fwhm_em(self):
        return 0.51 * self._wavelength_em / self.na

    @property
    def _num_scan_steps(self):
        if self._periodicity is None:
            self._periodicity, self._increment = self.compute_periodicity()
        return int((self._periodicity / self._fwhm_ex) * np.sqrt(2))

    @property
    def _scan_step(self):
        if self._periodicity is None:
            self._periodicity, self._increment = self.compute_periodicity()
        return self._periodicity / self._num_scan_steps

    ##### Main Methods #####
    def _check_input(self, img: torch.Tensor) -> torch.Tensor:
        # Shape check
        if img.ndim == 2:
            new_img = img.unsqueeze(0).unsqueeze(0)
        elif img.ndim == 3:
            new_img = img.unsqueeze(0)
        elif img.ndim == 4:
            new_img = img
        else:
            raise ValueError(
                f"Number of dimensions {img.ndim} is not compatible with the microscope. It's either not an image or too complex."
            )

        # Depth check (if image depth different than self._num_slices, we either throw an error or keep the center region)
        if new_img.shape[1] > self._num_slices:
            center_slice = new_img.shape[1] // 2
            starting_slice = center_slice - (self._num_slices // 2)
            new_img = new_img[
                :, starting_slice : starting_slice + self._num_slices, ...
            ]
        elif new_img.shape[1] < self._num_slices:
            raise ValueError(
                f"Tensor depth {new_img.shape[1]} is less than the number of slices of the microscope {self._num_slices}. Please, add more layers or change the microscope configuration."
            )

        # Type check
        if new_img.dtype == torch.uint8:
            new_img = new_img.to(dtype=torch.float32)
            new_img /= 255
        elif new_img.dtype in (float, torch.float16, torch.float64):
            new_img = new_img.to(dtype=torch.float32)
        elif new_img.dtype == torch.float32:
            new_img = new_img
        else:
            raise ValueError(
                f"Tensor type {new_img.dtype} is not compatible with the microscope, it must be one of {(torch.uint8, float, torch.float16, torch.float32, torch.float64)}."
            )

        return new_img

    def get_num_slices(self) -> int:
        return self._num_slices

    def compute_periodicity(self) -> tuple[float, float]:
        incr0 = self._periodicity_raw / self._pix_size_simu
        divisors = [
            d
            for d in range(1, self._cam_size_simu[0] + 1)
            if self._cam_size_simu[0] % d == 0
        ]
        incr = min(divisors, key=lambda d: abs(d - incr0))
        periodicity_new = incr * self._pix_size_simu
        return periodicity_new, incr

    def generate_psf(self) -> tuple[torch.Tensor, torch.Tensor]:
        psf_ex_ft = calc_utils.generate_psf(
            self._num_slices,
            self._cam_size_simu[0],
            self._cam_size_simu[1],
            self._pix_size_simu * 1e-6,
            self._pix_size_simu_axi * 1e-6,
            self._wavelength_ex * 1e-6,
            self.na,
            self.device,
        )
        psf_ex_ifft = torch.fft.ifftshift(
            torch.fft.ifftn(torch.sqrt(psf_ex_ft), dim=(-2, -1)), dim=(-2, -1)
        )

        psf_em_ft = calc_utils.generate_psf(
            self._num_slices,
            self._cam_size_simu[0],
            self._cam_size_simu[1],
            self._pix_size_simu * 1e-6,
            self._pix_size_simu_axi * 1e-6,
            self._wavelength_em * 1e-6,
            self.na,
            self.device,
        )
        return psf_ex_ifft, psf_em_ft

    def generate_pattern(self) -> tuple[torch.Tensor, int]:
        holos_x, holos_y, intensity_x, intensity_y, n_steps_x, n_steps_y = (
            calc_utils.create_holograms(
                pattern_type=self.pattern_case,
                H_pix=self._cam_size_simu[0],
                W_pix=self._cam_size_simu[1],
                incr=self._increment,
                n_steps=self._num_scan_steps,
                phase_optim=self.phase_optim,
                n_aods=self._num_aods,
                device=self.device,
                iterations=150,  # Hardcoded
            )
        )
        holos_x = holos_x
        holos_y = holos_y
        holo = torch.outer(holos_y[0], holos_x[0]).to(device=self.device)
        n_steps = torch.tensor([n_steps_x, n_steps_y], device=self.device)

        roll_step = 16 if self.aod_propag == 1 else self._cam_size_simu[0]
        num_rolls = self._cam_size_simu[0] // roll_step
        roll_direction = (1, 1) if self._num_aods == 2 else (1, 0)

        calib_pattern = torch.zeros(
            (self._num_slices, *self._cam_size_simu),
            dtype=torch.float32,
            device=self.device,
        )

        for r in range(num_rolls):
            if not r % (num_rolls // 10):
                print(f"Computing Excitation {100 * (r + 1) / num_rolls:.1f} %")
            holo_rolled = torch.roll(
                holo, shifts=(r * roll_direction[0], r * roll_direction[1]), dims=(0, 1)
            )

            for z in range(self._num_slices):
                product = self.psf_ex[z] * holo_rolled
                calib_pattern += torch.abs(torch.fft.fftn(product, dim=(-2, -1))) ** 2

            del holo_rolled

        calib_pattern /= num_rolls
        calib_pattern /= torch.max(calib_pattern)
        return calib_pattern, n_steps

    def noisy_reading(self, img: torch.Tensor) -> torch.Tensor:
        large_img = F.interpolate(
            img,
            size=(self._cam_size_simu[0], self._cam_size_simu[1]),
            mode="bicubic",
            align_corners=True,
        )

        physics_upsampling_gaussian = dinv.physics.Upsampling(
            large_img.shape[1:],
            filter="bicubic",
            factor=self.upsampling_factor,
            device=self.device,
        )
        physics_upsampling_gaussian.set_noise_model(
            dinv.physics.GaussianNoise(self.noise_level_gaussian)
        )
        defocused_img = physics_upsampling_gaussian(large_img)

        physics_inpainting_poisson = dinv.physics.Inpainting(
            defocused_img.shape[1:], mask=self.inpainting_mask, device=self.device
        )
        physics_inpainting_poisson.set_noise_model(
            dinv.physics.PoissonNoise(self.noise_level_poisson, clip_positive=True)
        )
        noisy_img = physics_inpainting_poisson(defocused_img)

        return noisy_img

    def process_img(self, img: torch.Tensor):
        img = self._check_input(img)
        B, D, H, W = img.shape

        noisy_img = self.noisy_reading(img)

        nonzero_slices = [
            k for k in range(self._num_slices) if torch.any(noisy_img[:, k, ...] != 0)
        ]

        pad_h = self._cam_size_simu[0] // 16
        pad_w = self._cam_size_simu[1] // 16

        output_stack = []
        calibs = []

        for ix in range(self._num_steps[0]):
            for iy in range(self._num_steps[1]):
                shift_x = (self._scan_step / self._pix_size_simu) * ix
                shift_y = (self._scan_step / self._pix_size_simu) * iy

                em: torch.Tensor = torch.zeros(
                    [
                        B,
                        self._cam_size_simu[0] + 2 * pad_h,
                        self._cam_size_simu[1] + 2 * pad_w,
                    ],
                    device=self.device,
                )

                for k in nonzero_slices:
                    calib_k = calc_utils.shift_fourier2D(
                        self._calib_pattern[k], (shift_x, shift_y)
                    )
                    product = noisy_img[:, k, ...] * calib_k

                    product_pad = F.pad(product, (pad_w, pad_w, pad_h, pad_h))
                    psf_pad = F.pad(
                        self.psf_em[k], (pad_w, pad_w, pad_h, pad_h)
                    ).unsqueeze(0)

                    em += torch.fft.fftshift(
                        torch.fft.irfft2(
                            torch.fft.rfft2(product_pad) * torch.fft.rfft2(psf_pad)
                        ).abs(),
                        dim=(1, 2),
                    )

                    if k == self._num_slices // 2:
                        calibs.append(calib_k)

                em_crop = em[..., pad_h:-pad_h, pad_w:-pad_w]
                donwsampling_physics = dinv.physics.DownsamplingMatlab(self.binn_simu)
                downsampled_output = donwsampling_physics(em_crop)
                output_stack.append(downsampled_output)

        output_stack = torch.stack(output_stack, dim=1).to(device=self.device)
        output_stack /= torch.max(output_stack)
        calibs = torch.stack(calibs).to(device=self.device)
        calibs /= torch.max(calibs)
        return output_stack, calibs

    def __call__(self, img: torch.Tensor):
        return self.process_img(img)


def main(args):
    import cv2

    # Process Micr Args
    micr_args = {
        k: v
        for k, v in vars(args).items()
        if k in inspect.getfullargspec(Microscope.__init__).args
    }

    test_img = "data/GTs/GT1.png"
    img = cv2.imread(test_img)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = torch.from_numpy(img).to(dtype=torch.float32, device="cuda") / 255
    physics = dinv.physics.GaussianNoise(0.2)
    img_z = physics(img).repeat(4, 1, 1)
    img = img.unsqueeze(0)
    img = torch.concat([img_z, img, img_z], dim=0)
    img = img.unsqueeze(0).contiguous().to(device="cuda")
    print(f"{img.shape=}")

    m = Microscope(**micr_args)
    output, calibs = m.process_img(img)
    print(f"{output.shape=}")
    print(f"{calibs.shape=}")

    dinv.utils.plot(
        {
            "GT": img[0, 4:5],
            "Microscope Output": output[0, :1],
            "Microscope Calibs": calibs[:1],
        }
    )
