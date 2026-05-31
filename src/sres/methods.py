import torch
import torch.nn.functional as F
from PIL import Image
from skimage.restoration import richardson_lucy
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
from torchvision.transforms import Grayscale

from src.utils.new_process_utils import jRL_process_fast

@torch.no_grad()
def swin2sr_x2(image: torch.Tensor, state):
    device = state.simulation_pipeline.microscope.device

    processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
    model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64").to(device=device)

    output_tensor = []
    for im in image:
        inputs = processor(Image.fromarray(im[0].detach().cpu().numpy()), return_tensors="pt").to(device=device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        output_tensor.append(Grayscale()(outputs.reconstruction.data.squeeze()).float().cpu().clamp_(0, 1))
    output_tensor = torch.stack(output_tensor, dim=0)
    if output_tensor.ndim == 3:
        output_tensor = output_tensor[None, ...]

    return output_tensor

@torch.no_grad()
def swin2sr_iterative_x2(image: torch.Tensor, state):
    device = state.simulation_pipeline.microscope.device

    processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
    model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64").to(device=device)

    output_tensor = []
    for im in image:
        output_single = []
        for ch in im:
            inputs = processor(Image.fromarray(ch.detach().cpu().numpy()), return_tensors="pt").to(device=device)

            with torch.no_grad():
                outputs = model(**inputs)
            
            output = Grayscale()(outputs.reconstruction.data.squeeze()).float().cpu().clamp_(0, 1)
            print(output.shape)
            output_single.append(output)
        output_tensor.append(torch.concatenate(output_single, dim=0))
    output_tensor = torch.stack(output_tensor, dim=0)

    if output_tensor.ndim == 3:
        output_tensor = output_tensor[None, ...]

    return output_tensor

@torch.no_grad()
def richardson_x1(image: torch.Tensor, state):
    psf = state.simulation_pipeline.microscope.psf_em[0][None, None, ...]
    
    kwargs = {
        "psf": F.interpolate(psf, scale_factor=image.shape[2]/psf.shape[2], mode="bilinear", align_corners=False, antialias=True).detach().cpu().numpy()[0, ...]
    }
    print(f"PSF shape: {kwargs['psf'].shape}")

    img = torch.mean(image, axis=1)
    output = richardson_lucy(img.detach().cpu().numpy(), **kwargs)[:, None, ...]
    output = torch.from_numpy(output)
    return output

@torch.no_grad()
def richardson_x2(image: torch.Tensor, state):
    kwargs = {
        "psf": state.simulation_pipeline.microscope.psf_em[0].detach().cpu().numpy()[None, ...]
    }
    print(f"PSF shape: {kwargs['psf'].shape}")

    img = torch.mean(image, axis=1)
    img = img[:, None, ...]
    img = F.interpolate(
        img, scale_factor=2, mode="bilinear", align_corners=False, antialias=True
    )[:, 0, ...]
    output = richardson_lucy(img.detach().cpu().numpy(), **kwargs)[:, None, ...]
    output = torch.from_numpy(output)
    return output

@torch.no_grad()
def jrl_x2(image: torch.Tensor, state):
    kwargs = {
        "psf": state.simulation_pipeline.microscope.psf_em[0],
        "exc_pat": torch.from_numpy(state.last_calib).to(device=image.device),
        "upsampling": 2,
    }
    print(f"PSF shape: {kwargs['psf'].shape}")
    image = image.to(device=state.simulation_pipeline.microscope.device)
    output = jRL_process_fast(image, **kwargs)
    output = output.detach().cpu()
    return output


ALGORITHMS = {
    "Richardson-Lucy x2": richardson_x2,
    "Richardson-Lucy x1": richardson_x1,
    "Joint Richardson-Lucy x2": jrl_x2,
    "Swin2SR Upsample x2": swin2sr_x2,
    "Swin2SR Upsample Iterative x2": swin2sr_iterative_x2,
}
