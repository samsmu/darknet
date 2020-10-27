#include "logistic_layer.h"
#include "activations.h"
#include "blas.h"
#ifdef GPU
#include "cuda.h"
#endif
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

layer make_logistic_layer(int batch, int inputs)
{
    printf("logistic x entropy                             %4d\n",  inputs);
    layer l = {(LAYER_TYPE)0 };
    l.type = LOGXENT;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = (float*)xcalloc(inputs * batch, sizeof(float));
    l.output = (float*)xcalloc(inputs * batch, sizeof(float));
    l.delta = (float*)xcalloc(inputs * batch, sizeof(float));
    l.cost = (float*)xcalloc(1, sizeof(float));

    l.forward = forward_logistic_layer;
    l.backward = backward_logistic_layer;

 #ifdef GPU
    l.forward_gpu = forward_logistic_layer_gpu;
    l.backward_gpu = backward_logistic_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.loss_gpu = cuda_make_array(l.loss, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
 #endif
    return l;
}

void forward_logistic_layer(const layer l, network_state state)
{
    copy_cpu(l.outputs*l.batch, state.input, 1, l.output, 1);
    activate_array(l.output, l.outputs*l.batch, LOGISTIC);
    //float temp = 0;
    if(state.truth){
        logistic_x_ent_cpu(l.batch*l.inputs, l.output, state.truth, l.delta, l.loss); //nithi loss function
	    int i;
	    int n = l.batch*l.inputs;
	    for(i = 0; i < n; ++i){
	        float t = state.truth[i];
	        float p = l.output[i];
                    printf("t %f,p %.10lf \n", t, p);
                    printf("err %.10lf, del %.10lf \n", l.loss[i], l.delta[i]);
	}             
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);;
        printf("my cost is %f \n", (l.cost[0]));
        
    }
}

void backward_logistic_layer(const layer l, network_state state)
{
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, state.delta, 1);
}

#ifdef GPU

void forward_logistic_layer_gpu(const layer l, network_state state)
{
    copy_ongpu(l.outputs*l.batch, state.input, 1, l.output_gpu, 1);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, LOGISTIC);
    if(state.truth ){
        logistic_x_ent_gpu(l.batch*l.inputs, l.output_gpu, state.truth, l.delta_gpu, l.loss_gpu);
        cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs); //sum_array_dice(l.loss,l.loss_temp,l.loss_temp1, l.batch*l.inputs);*/
    }
}

void backward_logistic_layer_gpu(const layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, state.delta, 1);
}

#endif
