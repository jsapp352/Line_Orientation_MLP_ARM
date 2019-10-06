#include <iostream>
#include <Eigen/Dense>
#include <vector>

#define LEARNING_RATE 1

using namespace Eigen;
using namespace std;


class NeuronLayer
{
    private:
        int neuron_count;
        int inputs_per_neuron;
        float max_weight;
        MatrixXf synaptic_weights;
    
    public:
        NeuronLayer(int neuron_count, int inputs_per_neuron, float max_weight, float *starting_weights)
        :neuron_count{neuron_count}, inputs_per_neuron{inputs_per_neuron}, max_weight{max_weight}
        {
            synaptic_weights = Map<MatrixXf>(starting_weights, neuron_count, inputs_per_neuron);
        }

        int getNeuronCount()
        {
            return neuron_count;
        }

        int getInputsPerNeuron()
        {
            return inputs_per_neuron;
        }

        int getMaxWeight()
        {
            return max_weight;
        }

        MatrixXf getSynapticWeights()
        {
            return synaptic_weights;
        }

        void adjust_weights(MatrixXf adjustments)
        {
            synaptic_weights += (adjustments * LEARNING_RATE);

            float maxAbsWeight = synaptic_weights.array().abs().maxCoeff();
            if (maxAbsWeight > max_weight)
                synaptic_weights *= (max_weight / maxAbsWeight);
        }
};

class NeuralNetwork
{
    private:
        vector<NeuronLayer> neuron_layers;

    protected:

        float sigmoid(float x)
        {
            return 1 / (1 + exp(-x));
        }

        float sigmoid_derivative(float x)
        {
            return x * (1 - x);
        }

        float activation_function(float x)
        {
            return sigmoid(x);
        }

        float activation_derivative(float x)
        {
            return sigmoid_derivative(x);
        }



    public:
        NeuralNetwork(vector<NeuronLayer> neuron_layers)
        :neuron_layers{neuron_layers}
        { }



};

int main()
{    
    float *starting_weights[] = 
    {
        new float[16]
        { 
             0.78693466,  0.00260829, -0.52798853,  0.92084359,
             0.85578172,  0.50035073,  0.14404296, -0.03027512,
            -0.43132144, -0.26774606, -0.89238543, -0.86610821,
             0.87852929,  0.08643093, -0.51677491, -0.03856048
        },            
        new float[16]
        {
            -0.36415303,  0.40740717,  0.50542251, -0.59614972,
             0.59996569, -0.69599694, -0.74805606,  0.61728582,
             0.50074242, -0.85998174, -0.33990181, -0.50978041,
            -0.92183461,  0.90926905, -0.44542875, -0.09638122
        },
        new float[12]
        {
            -0.11044091, -0.67848051, -0.70275169,
            -0.65093601,  0.4998441,   0.95182468,
            -0.62298891, -0.00173666,  0.59165978,
            -0.72568407,  0.90145892,  0.39221916
        }
    };

    //DEBUG Begin test code:

    NeuronLayer layer0{3, 4, 10.0, starting_weights[2]};

    cout << "Layer 0 has " << layer0.getNeuronCount() << " neurons, each with ";
    cout << layer0.getInputsPerNeuron() << " inputs." << endl << endl;

    cout << "Layer 0 starting synaptic weights: " << endl << layer0.getSynapticWeights() << endl;

    MatrixXf adjustments(3, 4);
    adjustments << 1, 0, 0, 1,
                    0, 1, 1, 0,
                    5, 5, 5, 5;
    
    layer0.adjust_weights(adjustments);

    cout << "Layer 0 adjusted synaptic weights: " << endl << layer0.getSynapticWeights() << endl;

    //DEBUG End test code.


    return 0;
}