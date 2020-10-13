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
    fprintf(stderr, "logistic x entropy                             %4d\n",  inputs);
    layer l = {0};
    l.type = LOGXENT;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = calloc(inputs*batch, sizeof(float));
    l.loss_temp1 = calloc(inputs*batch, sizeof(float));
    l.loss_temp = calloc(inputs*batch, sizeof(float));
    l.output = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));

    l.forward = forward_logistic_layer;
    l.backward = backward_logistic_layer;

 #ifdef GPU
    l.forward_gpu = forward_logistic_layer_gpu;
    l.backward_gpu = backward_logistic_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch); 
    l.loss_gpu = cuda_make_array(l.loss, inputs*batch); 
    //l.loss_gpu_temp = cuda_make_array(l.loss, inputs*batch); 
    //l.loss_gpu_temp1 = cuda_make_array(l.loss, inputs*batch); 
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch); 
 #endif
    return l;
}

void forward_logistic_layer(const layer layer, network_state state)
{
    copy_cpu(layer.outputs*layer.batch, state.input, 1, layer.output, 1);
    activate_array(layer.output, layer.outputs*layer.batch, LOGISTIC);
    //float temp = 0;
    if(state.truth){
        logistic_x_ent_cpu(layer.batch*layer.inputs, layer.output, state.truth, layer.delta, layer.loss); //nithi loss function
	    int i;
	    int n = layer.batch*layer.inputs;
	    for(i = 0; i < n; ++i){
	        float t = state.truth[i];
	        float p = layer.output[i];
                    printf("t %f,p %.10lf \n", t, p);
                    printf("err %.10lf, del %.10lf \n", layer.loss[i], layer.delta[i]);
	}             
        layer.cost[0] = sum_array(layer.loss, layer.batch*layer.inputs);;
        printf("my cost is %f \n", (layer.cost[0]));
        
    }
}

void backward_logistic_layer(const layer layer, network_state state)
{
    axpy_cpu(layer.inputs*layer.batch, 1, layer.delta, 1, state.delta, 1);
}

#ifdef GPU

void forward_logistic_layer_gpu(const layer layer, network_state state)
{
    copy_ongpu(layer.outputs*layer.batch, state.input, 1, layer.output_gpu, 1);
    activate_array_gpu(layer.output_gpu, layer.outputs*layer.batch, LOGISTIC);
    if(state.truth){
        logistic_x_ent_gpu(layer.batch*layer.inputs, layer.output_gpu, state.net.truth_gpu, layer.delta_gpu, layer.loss_gpu);
        cuda_pull_array(layer.loss_gpu, layer.loss, layer.batch*layer.inputs);
        layer.cost[0] = sum_array(layer.loss, layer.batch*layer.inputs); //sum_array_dice(l.loss,l.loss_temp,l.loss_temp1, l.batch*l.inputs);
    }
}

void backward_logistic_layer_gpu(const layer layer, network_state state)
{
    axpy_ongpu(layer.batch*layer.inputs, 1, layer.delta_gpu, 1, state.net.delta_gpu, 1);
}

#endif
