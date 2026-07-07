#include <torch/extension.h>
void _gs_render(
        const torch::Tensor &sigmas,
	const torch::Tensor &coords,
	const torch::Tensor &colors,
	torch::Tensor &rendered_img,
	const int s, 
	const int h, 
	const int w,
	const int c,
	const float dmax
);

void _gs_render_backward(
        const torch::Tensor& sigmas,
	const torch::Tensor& coords,
	const torch::Tensor& colors,
	const torch::Tensor& grads, // (h, w, c)
	torch::Tensor& grads_sigmas,
	torch::Tensor& grads_coords,
	torch::Tensor& grads_colors,
	const int s, 
	const int h, 
	const int w,
	const int c,
	const float dmax
);