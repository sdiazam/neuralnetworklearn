use std::{f32, iter::once};
use rand::{Rng, RngCore};

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
}

#[derive(Debug)]
pub struct LayerTopology {
    pub neurons: usize,
}

impl Network {
    // returns a Network, layers is a vector which contains the number of neurons in that layer
    pub fn random(rng: &mut dyn RngCore, layers: &[LayerTopology]) -> Self{

        assert!(layers.len() > 1); // doesn't make sense to have a network with 1 layer

        // for layers[i] and layers[i+1] essentially passing each output as the input for the next layer
        let layers = layers
            .windows(2)
            .map(|layers| Layer::random(rng,layers[0].neurons, layers[1].neurons))
            .collect();

        Self { layers }
    }

    // propagation is, take inputs, multiply by weights, add bias, then rotate/move
    // we use mut inputs because inputs is being modified by each layer with propagate
    // &self is a reference to a network
    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {

        // instead of using a for loop we can use iterate() and fold() and just return the final result of this
        self.layers
            .iter()// go through each layer in order.
            .fold(inputs, |inputs,layer| layer.propagate(inputs))
            //.fold(initial_value, closure) = repeatedly apply propagate, accumulating the result. 
            // initial_value = the first input vector 
            // closure = at each step: layer.propagate(inputs)
    }

    pub fn weights(&self) -> impl Iterator<Item = f32> + '_ {
        self.layers
            .iter()// iterate layers
            .flat_map(|layer| layer.neurons.iter())
            .flat_map(|neuron| once(&neuron.bias).chain(&neuron.weights))
            .copied()
    }

    pub fn from_weights(
        layers: &[LayerTopology],
        weights: impl IntoIterator<Item = f32>,
    ) -> Self {
        assert!(layers.len() > 1);

        let mut weights = weights.into_iter();

        let layers = layers
            .windows(2)
            .map(|layers| {
                Layer::from_weights(
                    layers[0].neurons,
                    layers[1].neurons,
                    &mut weights,
                )
            })
            .collect();

        if weights.next().is_some() {
            panic!("got too many weights");
        }

        Self { layers }
    }


}

#[derive(Debug)]
struct Layer{
    neurons: Vec<Neuron>,
}

impl Layer {

    fn random(rng: &mut dyn RngCore, input_size: usize, output_size: usize)-> Self {
        let neurons = (0..output_size)
            .map(|_| Neuron::random(rng, input_size))// argument is _ as it is not important
            .collect(); // collect in vector of 

        Self { neurons }
    }

    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        // outputs are a single number, all will be stored in a vector for the Layer to propagate
        self.neurons
            .iter()
            .map(|neuron| neuron.propagate(&inputs)) // compute each neurons output based on one input
            .collect() // collect all outputs into a vector
    }

    fn from_weights(
        input_size: usize,
        output_size: usize,
        weights: &mut dyn Iterator<Item = f32>,
    ) -> Self {
        let neurons = (0..output_size)
            .map(|_| Neuron::from_weights(input_size, weights))
            .collect();

        Self { neurons }
    }
}

#[derive(Debug)]
struct Neuron{
    bias: f32,
    weights: Vec<f32>
}

impl Neuron {

    fn random(rng: &mut dyn RngCore, input_size: usize) -> Self {
        // random bias
        let bias = rng.gen_range(-1.0..=1.0);

        let weights = (0..input_size)
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();

        Self { bias, weights }
    }

    // return a single value
    // work with borrowed values so that input in layer.propagate never has to be moved
    fn propagate(&self, inputs: &[f32]) -> f32 {
        assert_eq!(inputs.len(), self.weights.len());

        // no need for indexing input or getting weight, zip, essentially makes tuples with both and creates an iterator in it for each
        let output = inputs
            .iter()// iterate
            .zip(&self.weights)// zip with weights (weight and inputs indexed in a tuple together)
            .map(|(input, weight)| input * weight)// map after multiplying
            .sum::<f32>(); // cannot infer type for sum() so we let it know ::<f32>

        (self.bias + output).max(0.0) // if 0..., else return it
    }

    fn from_weights(
        input_size: usize,
        weights: &mut dyn Iterator<Item = f32>,
    ) -> Self {
        let bias = weights.next().expect("got not enough weights");

        let weights = (0..input_size)
            .map(|_| weights.next().expect("got not enough weights"))
            .collect();

        Self { bias, weights }
    }
}








//tests
#[cfg(test)]
mod tests {
    use std::vec;

    use super::*; // use neural network
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use approx::assert_relative_eq;

    #[test]
    fn random() {
        // Because we always use the same seed, our `rng` in here will
        // always return the same set of values
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let neuron = Neuron::random(&mut rng, 4);
        
        // we need the approximate eq, because f32 are not entirely accurate
        assert_relative_eq!(neuron.bias, -0.6255188);

        assert_relative_eq!(
            neuron.weights.as_slice(),
            [0.67383957, 0.8181262, 0.26284897, 0.5238807].as_ref()
        );
    }

    #[test]
    fn propagate() {
        // define our neuron
        let neuron = Neuron {
            bias: 0.5, // little nudge to activate, even if motivation is not much
            weights: vec![-0.3, 0.8], // doesn't like left food much, likes right food a lot
        };

        // Ensures `.max()` (our ReLU) works:
        // ensure that with the very very far food, activation will go to 0
        assert_relative_eq!( neuron.propagate(&[-10.0, -10.0]), 0.0); // inputs of -10, output will be roughly -4.5 after propagating
        // output should be 1.15 on l, we write the full equation for clarity
        // weights and inputs should be getting multiplied 
        // (i like food here this much * this is how far away it is) + a little nudge to ensure movement even on very small outputs
        assert_relative_eq!(neuron.propagate(&[0.5, 1.0]) , (-0.3 * 0.5) + (0.8 * 1.0) + 0.5); 
    }

    #[test]
    fn layer_propagate() {
        let layer = Layer {
            neurons: vec![
                Neuron {
                    weights: vec![0.5, 0.5], // likes left and right equally
                    bias: 0.0, // no push
                },
                Neuron {
                    weights: vec![1.0, -1.0], // likes left, doesn't like right
                    bias: 1.0,
                },
            ],
        };// define layer

        let inputs = vec![0.6,0.2]; // higher means closer

        let outputs = layer.propagate(inputs); // apply the function to the neurons

        
        // First neuron: (0.6 * 0.5) + (0.2 * 0.5) + 0.0 = 0.3 + 0.1 = 0.4
        assert_relative_eq!(outputs[0], 0.4);

        // Second neuron: (0.6 * 1.0) + (0.2 * -1.0) + 1.0 = 0.6 - 0.2 + 1.0 = 1.4
        assert_relative_eq!(outputs[1], 1.4);
        
    }

    #[test]
    fn network_propagate() {
        
        let network = Network {
            layers : 
            vec![Layer{// single layer
                neurons: vec![
                    Neuron{// sing neuron
                        weights: vec![0.5, 0.5], // likes left and right equally
                        bias: 0.0, // no push
                    }
                ]
            }]
        };
        
        let inputs = vec![0.6,0.2];
        let outputs = network.propagate(inputs);

        // expected result is 0.4 as the output is the output from the neuron propagate
        assert_relative_eq!(outputs[0], 0.4);

    }



}