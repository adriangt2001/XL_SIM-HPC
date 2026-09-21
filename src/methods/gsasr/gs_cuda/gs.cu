#include <stdio.h>   
#include <cmath>
#include <curand_kernel.h>

#define PI 3.1415926536
#define PI2 6.283153072
#define MAX_NUM_CHANNELS 32
#define BLOCK_SIZE 256

extern "C"
__global__ void _gs_render_cuda(
        const float *sigmas,
        const float *coords,
        const float *colors,
        float *rendered_img,
	const int s,  // gs num
	const int h, 
	const int w,
	const int c
	){

        int index = blockIdx.x*blockDim.x + threadIdx.x;
	int curw = index % w;
	int curh = int((index-curw)/w);
	if(curw >= w || curh >=h){
	    return;
	}

	float curw_f = 2.0*curw/(w-1) - 1.0;
	float curh_f = 2.0*curh/(h-1) - 1.0;

        // printf("index:%d, curw:%d, curh:%d, curw_f:%f, curh_f:%f\n",index,curw,curh,curw_f,curh_f);
	
	for(int si=0; si<s; si++){

	    // compute the 2d gs value
	    float sigma_x = sigmas[si*3+0];
	    float sigma_y = sigmas[si*3+1];
	    float rho = sigmas[si*3+2];
            float x = coords[si*2+0];
            float y = coords[si*2+1];

	    // 
            float one_div_one_minus_rho2 = 1.0 / (1-rho*rho) ;
            float one_div_sigma_x = 1.0 / sigma_x;
            float one_div_sigma_y = 1.0 / sigma_y;
	    float d_x = curw_f - x;
	    float d_y = curh_f - y;

            float v = one_div_sigma_x*one_div_sigma_x*d_x*d_x;
            v -= 2*rho*d_x*d_y*one_div_sigma_x*one_div_sigma_y;
            v += d_y*d_y*one_div_sigma_y*one_div_sigma_y;
            v *= -one_div_one_minus_rho2 / 2.0;
            v = expf(v);
	    // since we normlize the v with the max, we remove this step to obtain equal result
            // v *= one_div_sigma_x * one_div_sigma_y * pow(one_div_one_minus_rho2, 0.5) / PI2 ;
            // printf("si:%d, sigma_x: %f, sigma_y:%f, rho:%f, x:%f, y:%f, v:%f\n", si, sigma_x, sigma_y, rho, x,y,v);

            for(int ci=0; ci<c; ci++){
		rendered_img[(curh*w+curw)*c+ci] += v*colors[si*c+ci];
	    }
	}
}


void _gs_render(
        const float *sigmas,
        const float *coords,
        const float *colors,
        float *rendered_img,
	const int s, 
	const int h, 
	const int w,
	const int c
	) {

        int threads=BLOCK_SIZE;
		dim3 block(threads);
		dim3 grid((h * w + threads - 1) / threads);
        _gs_render_cuda<<<grid, block>>>(sigmas, coords, colors, rendered_img, s, h, w, c);
}

extern "C"
__global__ void _gs_render_backward_cuda(
        const float *sigmas,
        const float *coords,
        const float *colors,
        const float *grads,
        float *grads_sigmas,
        float *grads_coords,
        float *grads_colors,
	const int s,  // gs num
	const int h, 
	const int w,
	const int c
	){

        int curs = blockIdx.x*blockDim.x + threadIdx.x;
	if(curs >= s){
	    return ;
	}

	// obtain parameters of gs
	float sigma_x = sigmas[curs*3+0];
	float sigma_y = sigmas[curs*3+1];
	float rho = sigmas[curs*3+2];
    float x = coords[curs*2+0];
    float y = coords[curs*2+1];
	
	const float* cur_colors = colors + curs * c;
	float* cur_grads_colors = grads_colors + curs * c;

	__shared__ float shared_grad_colors[BLOCK_SIZE][MAX_NUM_CHANNELS];
	const bool use_shared = (c <= MAX_NUM_CHANNELS);

	float* grad_colors;

	if (use_shared) {
		grad_colors = shared_grad_colors[threadIdx.x];
		
		for (int ci = 0; ci < c; ++ci)
			grad_colors[ci] = 0.0f;
	}
	else {
		grad_colors = cur_grads_colors;
		for (int ci = 0; ci < c; ci++)
			grad_colors[ci] = 0.0f;
	}

	//
        float w1 = -0.5 / (1-rho*rho) ;
        float w2 = 1.0 / (sigma_x*sigma_x);
        float w3 = 1.0 / (sigma_x*sigma_y);
        float w4 = 1.0 / (sigma_y*sigma_y);
	float od_sx = 1.0 / sigma_x;
	float od_sy = 1.0 / sigma_y;

        // init 
	float _gx=0.0, _gy=0.0;
	float _gsx=0.0, _gsy=0.0, _gsr=0.0;

	for(int hi = 0; hi < h; hi++){
	    for( int wi=0; wi < w; wi++){

	        float curw_f = 2.0*wi/(w-1) - 1.0;
	        float curh_f = 2.0*hi/(h-1) - 1.0;

		// obtain grad to p^t_r, p^t_g, p^t_b
		const float* pixel_grad = grads + (hi * w + wi) * c; // grad of loss to P^t_r

	        // compute the 2d gs value

		float d_x = curw_f - x; // distance along x axis
		float d_y = curh_f - y;
                float d = w2*d_x*d_x - 2*rho*w3*d_x*d_y + w4*d_y*d_y;
		float v = w1*d;
		v = expf(v);
                // printf("si:%d, sigma_x: %f, sigma_y:%f, rho:%f, x:%f, y:%f, v:%f\n", si, sigma_x, sigma_y, rho, x,y,v);

		// compute grad of coords and colors
		float gpt = 0.0f;
		for (int ci = 0; ci < c; ci++) {
			float g = pixel_grad[ci];
			grad_colors[ci] += v * g;
			gpt += g * cur_colors[ci];
		}

		float v_2_w1 = v*2*w1;

		float g_vst_to_gsx = v_2_w1*(-w2*d_x+rho*w3*d_y); // grad of v^{st} to G^s_x
		_gx += gpt*g_vst_to_gsx;
		float g_vst_to_gsy = v_2_w1*(-w4*d_y+rho*w3*d_x); // grad of v^{st} to G^s_y
		_gy += gpt*g_vst_to_gsy;

		// compute grad of sigmas
		float g_vst_to_gsigx = v_2_w1*od_sx* (w3*rho*d_x*d_y - w2*d_x*d_x);
		_gsx += gpt*g_vst_to_gsigx;
		float g_vst_to_gsigy = v_2_w1*od_sy* (w3*rho*d_x*d_y - w4*d_y*d_y);
		_gsy += gpt*g_vst_to_gsigy;
		float g_vst_to_rho = -v_2_w1*(2*w1*rho*d+w3*d_x*d_y);
		_gsr += gpt*g_vst_to_rho;
	}
    }

    // write the values 
    grads_sigmas[curs*3+0] = _gsx;
    grads_sigmas[curs*3+1] = _gsy;
    grads_sigmas[curs*3+2] = _gsr;
    grads_coords[curs*2+0] = _gx;
    grads_coords[curs*2+1] = _gy;

	if (use_shared) {
		for (int ci = 0; ci < c; ++ci)
			cur_grads_colors[ci] = grad_colors[ci];
	}

}

void _gs_render_backward(
        const float *sigmas,
        const float *coords,
        const float *colors,
	const float *grads, // (h, w, c)
	float *grads_sigmas,
	float *grads_coords,
	float *grads_colors,
	const int s, 
	const int h, 
	const int w,
	const int c
	) {

        int threads=BLOCK_SIZE;
		dim3 block(threads);
        dim3 grid((s + threads - 1) / threads);
        _gs_render_backward_cuda<<<grid, block>>>(sigmas, coords, colors, grads, grads_sigmas, grads_coords, grads_colors, s, h, w, c);
}

