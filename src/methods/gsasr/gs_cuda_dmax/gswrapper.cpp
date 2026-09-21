#include "gs.h"
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor gs_render(
        const torch::Tensor &sigmas,
        const torch::Tensor &coords,
        const torch::Tensor &colors,
	const int64_t s,
	const int64_t h,
	const int64_t w,
	const int64_t c,
	const double dmax
        ){
      
        CHECK_INPUT(sigmas);
        CHECK_INPUT(coords);
        CHECK_INPUT(colors);

        // run the code at the cuda device same with the input
        const at::cuda::OptionalCUDAGuard device_guard(device_of(sigmas));
        
        torch::Tensor rendered_img = torch::zeros({h, w, c}, sigmas.options());

        _gs_render(
            sigmas,
            coords,
            colors,
            rendered_img,
	    s, h, w, c, static_cast<float>(dmax));

        return rendered_img;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gs_render_backward(
        const torch::Tensor &sigmas,
        const torch::Tensor &coords,
        const torch::Tensor &colors,
        const torch::Tensor &grads,
	const int64_t s,
	const int64_t h,
	const int64_t w,
	const int64_t c,
	const double dmax
        ){

        CHECK_INPUT(sigmas);
        CHECK_INPUT(coords);
        CHECK_INPUT(colors);
        CHECK_INPUT(grads);


        // run the code at the cuda device same with the input
        const at::cuda::OptionalCUDAGuard device_guard(device_of(sigmas));

        torch::Tensor grads_sigmas = torch::zeros_like(sigmas);
        torch::Tensor grads_coords = torch::zeros_like(coords);
        torch::Tensor grads_colors = torch::zeros_like(colors);

        _gs_render_backward(
            sigmas,
            coords,
            colors,
            grads,
            grads_sigmas,
            grads_coords,
            grads_colors,
	    s, h, w, c, static_cast<float>(dmax));
        
        return {grads_sigmas, grads_coords, grads_colors};
}

TORCH_LIBRARY(gscuda, m) {
        m.def("gs_render(Tensor sigmas, Tensor coords, Tensor colors, int s, int h, int w, int c, float dmax) -> Tensor");
        m.def("gs_render_backward(Tensor sigmas, Tensor coords, Tensor colors, Tensor grads, int s, int h, int w, int c, float dmax) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(gscuda, CUDA, m) {
        m.impl("gs_render", &gs_render);
        m.impl("gs_render_backward", &gs_render_backward);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
