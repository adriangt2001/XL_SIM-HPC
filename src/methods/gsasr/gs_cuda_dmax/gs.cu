#include <stdio.h>
#include <cmath>
#include <torch/extension.h>

#define PI 3.1415926536
#define PI2 6.283153072
#define MAX_NUM_CHANNELS 16
#define BLOCK_SIZE 256

template<typename scalar_t>
__global__ void _gs_render_cuda(
	const scalar_t *sigmas,
	const scalar_t *coords,
	const scalar_t *colors,
	scalar_t *rendered_img,
	const int s,  // gs num
	const int h, 
	const int w,
	const int c,
	const float dmax
)
{
	int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pixelIdx >= h*w) return;

	int hi = pixelIdx / w;
	int wi = pixelIdx % w;

	float curh_f = 2.0f * hi / (h - 1) - 1.0f;
	float curw_f = 2.0f * wi / (w - 1) - 1.0f;

	float accum[MAX_NUM_CHANNELS];

	for (int ci = 0; ci < c; ci++) {
		accum[ci] = 0.0f;
	}

	for (int si = 0; si<s; si++) {
		float x = static_cast<float>(coords[si*2+0]);
		float y = static_cast<float>(coords[si*2+1]);
		
		float d_x = curw_f - x;
		float d_y = curh_f - y;

		if ((d_y > dmax || d_y < -dmax) || (d_x > dmax || d_x < -dmax)) continue;

		float sigma_x = static_cast<float>(sigmas[si*3+0]);
		float sigma_y = static_cast<float>(sigmas[si*3+1]);
	    float rho = static_cast<float>(sigmas[si*3+2]);

		float one_div_one_minus_rho2 = 1.0 / (1-rho*rho);
		float one_div_sigma_x = 1.0 / sigma_x;
		float one_div_sigma_y = 1.0 / sigma_y;

		float v = one_div_sigma_x*one_div_sigma_x*d_x*d_x;
		v -= 2*rho*d_x*d_y*one_div_sigma_x*one_div_sigma_y;
		v += d_y*d_y*one_div_sigma_y*one_div_sigma_y;
		v *= -one_div_one_minus_rho2 / 2.0;
		v = expf(v);

		int color_base = si * c;

		for (int ci=0; ci < c; ci++) {
			float color = static_cast<float>(colors[color_base + ci]);
			accum[ci] += v * color;
		}
	}

	for (int ci = 0; ci < c; ci++) {
		rendered_img[(hi*w+wi) * c + ci] = static_cast<scalar_t>(accum[ci]);
	}
}

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
	) {

        int threads=BLOCK_SIZE;
        dim3 block(threads);
        dim3 grid((h * w + threads - 1) / threads);

		AT_DISPATCH_FLOATING_TYPES_AND2(
			at::kHalf,
			at::kBFloat16,
			sigmas.scalar_type(),
			"_gs_render",
			[&] {
				_gs_render_cuda<scalar_t><<<grid, block>>>(
					sigmas.data_ptr<scalar_t>(),
					coords.data_ptr<scalar_t>(),
					colors.data_ptr<scalar_t>(),
					rendered_img.data_ptr<scalar_t>(),
					s, h, w, c, dmax
				);
			}
        );
}

template <typename scalar_t>
__global__ void _gs_render_backward_cuda(
	const scalar_t *sigmas,
	const scalar_t *coords,
	const scalar_t *colors,
	const scalar_t *grads,
	scalar_t *grads_sigmas,
	scalar_t *grads_coords,
	scalar_t *grads_colors,
	const int s,  // gs num
	const int h, 
	const int w,
	const int c,
	const float dmax
)
{
	int curs = blockIdx.x * blockDim.x + threadIdx.x;
    if (curs >= s) return;

	float sigma_x = static_cast<float>(sigmas[curs*3 + 0]);
    float sigma_y = static_cast<float>(sigmas[curs*3 + 1]);
    float rho = static_cast<float>(sigmas[curs*3 + 2]);

    float x = static_cast<float>(coords[curs*2 + 0]);
    float y = static_cast<float>(coords[curs*2 + 1]);

	float w1 = -0.5f / (1.0f - rho*rho);
    float w2 = 1.0f / (sigma_x*sigma_x);
    float w3 = 1.0f / (sigma_x*sigma_y);
    float w4 = 1.0f / (sigma_y*sigma_y);

    float od_sx = 1.0f / sigma_x;
    float od_sy = 1.0f / sigma_y;

	float grad_coord_x = 0.0f;
    float grad_coord_y = 0.0f;

    float grad_sigma_x = 0.0f;
    float grad_sigma_y = 0.0f;
    float grad_rho = 0.0f;

	float grad_color[MAX_NUM_CHANNELS];
    for (int ci = 0; ci < c; ci++)
        grad_color[ci] = 0.0f;

    int color_base = curs * c;

	for (int hi = 0; hi < h; hi++) {

        float curh_f = 2.0f * hi / (h - 1) - 1.0f;
        float d_y = curh_f - y;

        if (fabsf(d_y) > dmax)
            continue;

        for (int wi = 0; wi < w; wi++) {

            float curw_f = 2.0f * wi / (w - 1) - 1.0f;
            float d_x = curw_f - x;

            if (fabsf(d_x) > dmax)
                continue;

            float d = w2*d_x*d_x
                    - 2.0f*rho*w3*d_x*d_y
                    + w4*d_y*d_y;

            float v = expf(w1*d);

            float v_2_w1 = v * 2.0f * w1;

            float g_vst_to_gsx =
                v_2_w1 * (-w2*d_x + rho*w3*d_y);

            float g_vst_to_gsy =
                v_2_w1 * (-w4*d_y + rho*w3*d_x);

            float g_vst_to_gsigx =
                v_2_w1 * od_sx *
                (rho*w3*d_x*d_y - w2*d_x*d_x);

            float g_vst_to_gsigy =
                v_2_w1 * od_sy *
                (rho*w3*d_x*d_y - w4*d_y*d_y);

            float g_vst_to_rho =
                -v_2_w1 *
                (2.0f*w1*rho*d + w3*d_x*d_y);

            int img_base = (hi*w + wi) * c;

            for (int ci = 0; ci < c; ci++) {

                float grad = static_cast<float>(grads[img_base + ci]);
                float color = static_cast<float>(colors[color_base + ci]);

                grad_color[ci] += v * grad;

                float tmp = grad * color;

                grad_coord_x += tmp * g_vst_to_gsx;
                grad_coord_y += tmp * g_vst_to_gsy;

                grad_sigma_x += tmp * g_vst_to_gsigx;
                grad_sigma_y += tmp * g_vst_to_gsigy;
                grad_rho     += tmp * g_vst_to_rho;
            }
        }
    }

    grads_coords[curs*2 + 0] = static_cast<scalar_t>(grad_coord_x);
    grads_coords[curs*2 + 1] = static_cast<scalar_t>(grad_coord_y);

    grads_sigmas[curs*3 + 0] = static_cast<scalar_t>(grad_sigma_x);
    grads_sigmas[curs*3 + 1] = static_cast<scalar_t>(grad_sigma_y);
    grads_sigmas[curs*3 + 2] = static_cast<scalar_t>(grad_rho);

    for (int ci = 0; ci < c; ci++)
        grads_colors[color_base + ci] = static_cast<scalar_t>(grad_color[ci]);
}

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
	) {

        int threads=BLOCK_SIZE;
        dim3 block(threads);
        dim3 grid((s + threads - 1) / threads);

		AT_DISPATCH_FLOATING_TYPES_AND2(
			at::kHalf,
			at::kBFloat16,
			sigmas.scalar_type(),
			"_gs_render_backward",
			[&] {
				_gs_render_backward_cuda<scalar_t><<<grid, block>>>(
						sigmas.data_ptr<scalar_t>(),
						coords.data_ptr<scalar_t>(),
						colors.data_ptr<scalar_t>(),
						grads.data_ptr<scalar_t>(),
						grads_sigmas.data_ptr<scalar_t>(),
						grads_coords.data_ptr<scalar_t>(),
						grads_colors.data_ptr<scalar_t>(),
						s, h, w, c, dmax
				);
			}
        );
}