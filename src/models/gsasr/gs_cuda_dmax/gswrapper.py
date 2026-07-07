import torch
import gscuda

def render(sigmas, coords, colors, rendered_img, dmax):
    h, w, c = rendered_img.shape
    s = sigmas.shape[0]
    rendered_img = torch.ops.gscuda.gs_render(sigmas, coords, colors, s, h, w, c, dmax)
    return rendered_img

def backward(ctx, grads):
    sigmas, coords, colors = ctx.saved_tensors
    dmax = ctx.dmax

    h, w, c = grads.shape
    s = sigmas.shape[0]

    grads_sigmas, grads_coords, grads_colors = torch.ops.gscuda.gs_render_backward(sigmas, coords, colors, grads, s, h, w, c, dmax)

    return grads_sigmas, grads_coords, grads_colors, None, None, None, None, None

def setup_context(ctx, inputs, output):
    sigmas, coords, colors, s, h, w, c, dmax = inputs
    ctx.save_for_backward(sigmas, coords, colors)
    ctx.dmax = dmax

torch.library.register_autograd("gscuda::gs_render", backward, setup_context=setup_context)

# def gaussiansplatting_render(sigmas, coords, colors, image_size,dmax=100):
#     sigmas = sigmas.contiguous() # (gs num, 3)
#     coords = coords.contiguous() # (gs num, 2)
#     colors = colors.contiguous() # (gs num, c)
#     h, w = image_size[:2]
#     c = colors.shape[-1]
#     rendered_img = torch.zeros(h, w, c).to(colors.device).to(torch.float32)
#     return GSCUDA.apply(sigmas, coords, colors, rendered_img, dmax)

# if __name__ == "__main__":
#     sigmas = torch.randn(10, 3).cuda()
#     coords = torch.randn(10, 2).cuda()
#     colors = torch.randn(10, 3).cuda()
#     image_size = (100, 100)
#     dmax = 0.1
#     rendered_img = gaussiansplatting_render(sigmas, coords, colors, image_size, dmax)
#     print(rendered_img.shape)