use std::ops::Index;
use rand::seq::IndexedRandom;
use rand::{Rng, RngCore};
// a trait means any type that implements this trait, must provide certain methods
// how to select what parents will pass their genes on
pub trait SelectionMethod { // any individual is able to use select because it is a trait of type I 
    fn select<'a, I>(&self, rng: &mut dyn RngCore, population: &'a [I]) -> &'a I // the returned reference is only valid as long as the population slice is valid.
    where
        I: Individual; // type 'I' MUST implement 'Individual' trait
}

// we begin with roulette wheel selection but i can try and implement rank selection afterwards
pub struct RouletteWheelSelection;

impl SelectionMethod for RouletteWheelSelection {
    fn select<'a, I>(&self, rng: &mut dyn RngCore, population: &'a [I]) -> &'a I
    where
        I: Individual,
    {
        population
            .choose_weighted(rng, |individual| individual.fitness())
            .expect("got an empty population")
    }
}

// method of passing over genes
pub trait CrossoverMethod {
    fn crossover(
        &self,
        rng: &mut dyn RngCore,
        parent_a: &Chromosome,
        parent_b: &Chromosome,
    ) -> Chromosome; // return one chromosome with genes from both
}

#[derive(Clone, Debug)]
pub struct UniformCrossover;

impl CrossoverMethod for UniformCrossover{
    fn crossover(
            &self,
            rng: &mut dyn RngCore,
            parent_a: &Chromosome,
            parent_b: &Chromosome,
        ) -> Chromosome {
        
        assert_eq!(parent_a.len(), parent_b.len());

        // randomly select a or b gene for all genes and push to child gene vector
        parent_a
            .iter()// iterate, zip with b iterator, map and in closing: choose one to collect
            .zip(parent_b.iter())
            .map(|(&a, &b)| if rng.random_bool(0.5) { a } else { b })
            .collect() // collect all into a vector
    }
}

pub trait MutationMethod {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome);
}

#[derive(Clone, Debug)]
pub struct GaussianMutation{
    // 0.0 = no genes will be touched
    // 1.0 = all genes will be touched
    chance: f32,

    // magnitude of the change
    // 0.0 = touched genes will not be modified
    // 3.0 = touched genes will be += or -= by at most 3.0
    coeff: f32,
}

impl GaussianMutation {
    pub fn new(chance: f32, coeff: f32) -> Self{
        assert!(chance >= 0.0 && chance <= 1.0); // chance must be greater than 0 and less than 1
        Self { chance, coeff }
    }
}

impl MutationMethod for GaussianMutation {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome) {
        // for all genes in child
        for gene in child.iter_mut(){
            // determine a sign for the mutation
            let sign = if rng.random_bool(0.5) {-1.0} else {1.0};

            // (chance of being true) if true multiply by magnitude * a sign, and multiply by a random decimal
            if rng.random_bool(self.chance as f64) {
                *gene += sign * self.coeff * rng.random::<f32>();
            }
        }
    }
}

pub trait Individual {
    fn fitness(&self) -> f32; // single fitness score
    fn chromosome(&self) -> &Chromosome; // chromosome object with genes list
    fn create(chromosome: Chromosome) -> Self;
}

#[derive(Clone, Debug)]
pub struct Chromosome {
    genes: Vec<f32>,
}

impl Chromosome {
    //get length
    pub fn len(&self) -> usize {
        self.genes.len()
    }
    // return iterator
    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.genes.iter()
    }
    // return mutable iterator
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.genes.iter_mut()
    }
}

impl Index<usize> for Chromosome {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.genes[index]
    }
} //instead of exposing genes, we can index Chromosome, each f32 representing a gene

impl FromIterator<f32> for Chromosome {
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        Self {
            genes: iter.into_iter().collect(),
        }
    }
} // easily collect genes into a chromosome

impl IntoIterator for Chromosome {
    type Item = f32;
    type IntoIter = std::vec::IntoIter<f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.genes.into_iter()
    }
} // easily iterate over genes in a chromosome

pub struct GeneticAlgorithm<S> {
    selection_method: S,
    crossover_method: Box<dyn CrossoverMethod>,
    mutation_method: Box<dyn MutationMethod>,
}

impl<S> GeneticAlgorithm<S>
where
    S: SelectionMethod,
{
    // 'static means that the crossover method cannot use any references that would become invalid
    pub fn new(
            selection_method: S, 
            crossover_method: impl CrossoverMethod + 'static,
            mutation_method: impl MutationMethod + 'static
        ) -> Self {
        Self { selection_method, crossover_method: Box::new(crossover_method), mutation_method: Box::new(mutation_method)}
    }

    pub fn evolve<I>(&self, rng: &mut dyn RngCore, population: &[I]) -> Vec<I>
    where
        I: Individual,
    {
        (0..population.len())// whole population
            .map(|_| {
                // choose two parents, select their fitness score
                let parent_a = self.selection_method.select(rng, population).chromosome();
                let parent_b = self.selection_method.select(rng, population).chromosome();

                // crossover their genes
                let mut child = self.crossover_method.crossover(rng, parent_a, parent_b);

                self.mutation_method.mutate(rng, &mut child);
                
                I::create(child)
            })
            .collect() // collect in a new vector
    }
}
























#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::BTreeMap;
    use std::iter::FromIterator;

    #[derive(Clone, Debug, PartialEq)]
    enum TestIndividual {
        /// For tests that require access to the chromosome
        WithChromosome { chromosome: Chromosome },

        /// For tests that don't require access to the chromosome
        WithFitness { fitness: f32 },
    }

    impl TestIndividual {
        fn new(fitness: f32) -> Self {
            Self::WithFitness { fitness }
        }
    }

    impl Individual for TestIndividual {
        fn create(chromosome: Chromosome) -> Self {
            Self::WithChromosome { chromosome }
        }

        fn chromosome(&self) -> &Chromosome {
            match self {
                Self::WithChromosome { chromosome } => chromosome,

                Self::WithFitness { .. } => {
                    panic!("not supported for TestIndividual::WithFitness")
                }
            }
        }

        fn fitness(&self) -> f32 {
            match self {
                Self::WithChromosome { chromosome } => {
                    chromosome.iter().sum()

                    // ^ the simplest fitness function ever - we're just
                    // summing all the genes together
                }

                Self::WithFitness { fitness } => *fitness,
            }
        }
    }

    impl PartialEq for Chromosome {
        fn eq(&self, other: &Self) -> bool {
            approx::relative_eq!(self.genes.as_slice(), other.genes.as_slice())
        }
    }

    #[test]
    fn genetic_algorithm() {
        fn individual(genes: &[f32]) -> TestIndividual {
            TestIndividual::create(genes.iter().cloned().collect())
        }

        let mut rng = ChaCha8Rng::from_seed(Default::default());

        let ga = GeneticAlgorithm::new(
            RouletteWheelSelection,
            UniformCrossover,
            GaussianMutation::new(0.5, 0.5),
        );

        let mut population = vec![
            individual(&[0.0, 0.0, 0.0]),
            individual(&[1.0, 1.0, 1.0]),
            individual(&[1.0, 2.0, 1.0]),
            individual(&[1.0, 2.0, 4.0]),
        ];

        // We're running `.evolve()` a few times, so that the differences between the
        // input and output population are easier to spot.
        //
        // No particular reason for a number of 10 - this test would be fine for 5, 20 or
        // even 1000 generations - the only thing that'd change is the magnitude of the
        // difference between the populations.
        for _ in 0..10 {
            population = ga.evolve(&mut rng, &population);
        }

        let expected_population = vec![
            individual(&[0.44769490, 2.0648358, 4.3058133]),
            individual(&[1.21268670, 1.5538777, 2.8869110]),
            individual(&[1.06176780, 2.2657390, 4.4287640]),
            individual(&[0.95909685, 2.4618788, 4.0247330]),
        ];

        assert_eq!(population, expected_population);
    }

    #[test]
    fn roulette_wheel_selection() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());// fake rng

        let population = vec![
            TestIndividual::new(2.0),
            TestIndividual::new(1.0),
            TestIndividual::new(4.0),
            TestIndividual::new(3.0),
        ]; // population of fake individuals with fitness values
        
        let mut actual_histogram = BTreeMap::new();

        // select 1000 fitnesses we want to test if ones with higher fitness are chosen more often, not always or randomly
        for _ in 0..1000 {
            let fitness = RouletteWheelSelection
                .select(&mut rng, &population)// pick an individual
                .fitness() as i32; // cast to i32 since histograms do not support f32

            *actual_histogram
                .entry(fitness)// check if the entry is in the map
                .or_insert(0) += 1; // if not then insert 0 count, add 1 to each entry
        }

        let expected_histogram = BTreeMap::from_iter([
            // (fitness, how many times this fitness has been chosen)
            (1, 98),
            (2, 202),
            (3, 278),
            (4, 422), // the highest fitness value WAS chosen most often so our code is right
        ]);

        assert_eq!(actual_histogram, expected_histogram);
    }

    #[test]
    fn uniform_crossover(){
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let parent_a = (1..=100).map(|n| n as f32).collect();
        let parent_b = (1..=100).map(|n| -n as f32).collect();
        let child = UniformCrossover.crossover(&mut rng, &parent_a, &parent_b);
        
        // Number of genes different between `child` and `parent_a`
        let diff_a = child.iter().zip(parent_a).filter(|(c, p)| *c != p).count();

        // Number of genes different between `child` and `parent_b`
        let diff_b = child.iter().zip(parent_b).filter(|(c, p)| *c != p).count();

        assert_eq!(diff_a, 49);
        assert_eq!(diff_b, 51);
    }

    mod gaussian_mutation {

        use super::*;

        fn actual(chance: f32, coeff: f32) -> Vec<f32> {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let mut child = vec![1.0, 2.0, 3.0, 4.0, 5.0].into_iter().collect();

            GaussianMutation::new(chance, coeff).mutate(&mut rng, &mut child);

            child.into_iter().collect()
        }

        mod given_zero_chance {
            use approx::assert_relative_eq;

            fn actual(coeff: f32) -> Vec<f32> {
                super::actual(0.0, coeff)
            }
            mod and_zero_coefficient {
                use super::*;

                #[test]
                fn does_not_change_the_original_chromosome() {
                    let actual = actual(0.0);
                    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }

            mod and_nonzero_coefficient {
                use super::*;

                #[test]
                fn does_not_change_the_original_chromosome() {
                    let actual = actual(0.5);
                    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }
        }

        mod given_fifty_fifty_chance {
            mod and_zero_coefficient {
                #[test]
                fn does_not_change_the_original_chromosome() {
                    todo!();
                }
            }

            mod and_nonzero_coefficient {
                #[test]
                fn slightly_changes_the_original_chromosome() {
                    todo!();
                }
            }
        }

        mod given_max_chance {
            mod and_zero_coefficient {
                #[test]
                fn does_not_change_the_original_chromosome() {
                    todo!();
                }
            }

            mod and_nonzero_coefficient {
                #[test]
                fn entirely_changes_the_original_chromosome() {
                    todo!();
                }
            }
        }
    }
}
