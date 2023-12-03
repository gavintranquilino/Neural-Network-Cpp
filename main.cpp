#include <vector>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include <cmath>
using namespace std;

struct Connection 
{ // number of output connections to the neuron 
    double weight;
    double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

// ************** class Neuron **************
class Neuron 
{
public: // construct a connection
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    double sumDOW(const Layer &nextLayer) const;
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

    
private:
    double m_outputVal; // most important data
    double m_myIndex; // so that each neuron knows its own index in the layer

    // outputs to send for each of the neurons to next layer 
    // needs to store a double weight and double change in weight
    // so this is a new struct type connection
    vector<Connection> m_outputWeights; 
    
    // static keyword to call function without *this
    static double randomWeight(void) { return rand() / double(RAND_MAX); }

    // activation / transfer function that only used in neuron
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    double m_gradient;

    // for backprop, tunable params for update input weights
    static double eta; // [0.0, 1.0], overall net training rate
    static double alpha; // [0.0, n], multiplier of last weight change (momentum)
    /*
        n (eta) - overall new learning rate
            0.0 - slow learner
            0.2 - medium learner
            1.0 - reckless learner

        a (alpha) - momentum
            0.0 - no momemtum
            0.5 - moderate momentum        
    */
};

double Neuron::eta = 0.15; // overall net learning rate
double Neuron::alpha = 0.5 // momentum

void Neuron::updateInputWeights(Layer &prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

     for (unsigned neuron = 0; neuron < prevLayer.size(); ++neuron)
     {
        Neuron &neuron = prevLayer[neuron]; // neuron we modify in previous layer
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
            // Individual input, meagnified by gradient and train rate (eta)
            eta
            * neuron.getOutputVal()
            * m_gradient
            
            // Also momentum calculation = a fraction of the previous delta weight
            + alpha // momentum rate, multiplier of old change in weight from last training sample
            * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
        
     }
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;

    // Sum our contributions of the errors at the noes we feed

    // loop all the neurons in next layer
    for (unsigned neuron = 0; neuron < nextLayer.size() - 1; ++neuron)
    {
        sum += m_outputWeights[neuron].weight * nextLayer[neuron].m_gradient;
        return sum;
    }
}

void Neuron::calcHiddenGradients(Layer &nextLayer)
{
    // compare to sum of derivatives of weights of next layer
    // because no given target values

    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
    /* 
        difference target value is supposed to have, 
        and the actual value that it does have
        and then it multiplies that diff by
        derivative of its output val
        f'(a)(x-a)
    */
   double delta = targetVal - outputVal;
   m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x)
{
    // tanh : output range [-1.0, 1.0]
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
    // actual
    // d/dx (tanh x) = 1 - (tanh(x))^2
    
    // tanh approximation 1-x^2
    return 1.0 - (x * x);
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;

    /*
        Loop through all the neurons of the previous layer
        sum the prev layer outputs (our inputs)
        include the bias node from previous layer
    */
   for (unsigned neuron = 0; neuron < prevLayer.size; ++neuron)
   {
        sum += prevLayer[neuron].getOutputVal() * 
               prevLayer[neuron].m_outputWeights[m_myIndex].weight;
   }

    // Activation / transfer function
    // doesn't change anything in the object
    // only used on the neurons -> private func static
   m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) 
{
    for (unsigned connections = 0; connections < numOutputs; ++connections)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}

// ************** class Net **************
class Net
{
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;

private:
    // Structure of the actual network
    /* 2D vector of vector of Neurons
        Net -> has Layers -> has neurons  
    */
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum] 
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
};

void getResults(vector<double> &resultVals) const
{
    resultVals.clear(); // clear previous results

    // iterate through last output layer
    for (unsigned neuron = 0; neuron < m_layers.back().size() - 1; ++neuron)
    {
        resultVals.push_back(m_layers.back()[neuron].getOutputVal());
    }
}

void Net::backProp(const vector<double> &targetVals) 
{
    /* 
        Calculate overall net error (root mean square of output neuron errors)
        this is what we're trying to minimize based on how much error there was
        compared to the actual expected outputs

        loop through all the output neurons (exclude bias)
    */
    Layer &outputLayer = m_Layers.back();
    m_error = 0.0;
    for (unsigned neuron = 0; neuron < outputLayer.size() - 1; ++neuron)
    {
        double delta = targetVals[neuron] - outputLayer[neuron].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1; // average error squared
    m_error = sqrt(m_error); // RMS

    // For console output help, error indication of how well net has been doing 
    m_recentAverageError = 
        (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
        / (m_recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradients
    for (unsigned neuron = 0; neuron < outputLayer.size() - 1; ++neuron)
    {
        outputLayer[neuron].calcOutputGradients(targetVals[neuron]);
    }

    // Calculate gradients on hidden layers right to left
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0 ; --layerNum)
    {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned neuron = 0; neuron < hiddenLayer.size(); ++neuron)
        {
            hiddenLayer[neuron].calcHiddenGradients(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer,
    // update the connection weights, FOR ALL LAYERS

    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
    {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned neuron = 0; neuron < layer.size(); ++neuron)
        {
            layer[neuron].updateInputWeights(prevLayer);
        }
    }
}

void Net::feedForward(const vector<double> &inputVals)
{
    // we assert what we believe to be true, if it's not it runtime error
    assert(inputVals.size() == m_layers[0].size() - 1);

   // Assign (latch) input values into input neurons 
   // loop through every input value 
   for (unsigned i = 0; i < inputVals.size(); ++i)
   {
        m_layers[0][i].setOutputVal(inputVals[i]);
   }

   // Forward propagate
   // loop every hidden layer -> loop every neuron -> pls feed forward
   for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
   {
        Layer &prevLayer = m_layers[layerNum - 1];

        // avoid each layer's bias neuron, no propagating it
        for (unsigned neuronNum = 0; neuronNum < m_layers[layerNum].size() - 1: ++neuronNum)
        {
            m_layers[layerNum][neuronNum].feedForward(prevLayer); 
            /* 
                this feedForward() is defined as a method of Neuron, not Net
                Neuron can be friend of Net to access all the other Neurons
                but it makes more sense to pass a pointer to reference Previous Layer
                that just contains all the previous neurons
            */

        }
   }
}

Net::Net(const vector<unsigned> &topology)
{
    // Create all the Layers in Net
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
    {
        m_layers.push_back(Layer()); // append Layer objects in Net vector
                                    // appends empty layer

        // how many neurons in said layer?
        // some logic to differentiate which is output layer
        // condition: layerNum == topology.size() - 1 (if this is output layer)
        // then: numOutputs = 0
        // else: numOutputs = topology[layerNum + 1] (how many neurons in next layer)
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        // Create all the Neurons in each Layer
        // Add a bias neuron to the layer, <= to loop one last time for that bias
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
        {
            // go to most recently appended layer (right to left)
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            cout << "New neuron!" << endl;
        }

        // force bias node output value to 1.0. Last neuron created above ^
        m_layers.back().back().setOutputVal(1.0);
    }
}

int main()
{
    // topology struct
    /*
        Ex: {3, 2, 1} 
        3 neurons in input layer
        2 neurons in hidden layer
        1 neuron in output layer
    */
   vector<unsigned> topology;
   topology.push_back(3);
   topology.push_back(2);
   topology.push_back(1);

    // parameters: numLayers, numValues per net
    Net net(topology);

    // variable length array
    vector<double> inputVals, targetVals, resultVals; 

    // to train, feeding forward all the input values
    net.feedForward(inputVals); // some arr or struct inputVals
    net.backProp(targetVals); /* after feed fwd, tell net what 
                                // outputs SHOULD have been
                                   tune the weights on the nodes */

    // Post training, output results                               
    net.getResults(resultVals); 
    return EXIT_SUCCESS;
}
