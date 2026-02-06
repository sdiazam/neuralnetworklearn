use crate::*;

#[derive(Debug)]
pub struct Animal {
    pub(crate) position: na::Point2<f32>, // statically sized, two-dimensional column point
    pub(crate) rotation: na::Rotation2<f32>, // two dimensional rotation matrix
    pub(crate) speed: f32,
    pub(crate) eye: Eye,
    pub(crate) brain: Brain,
    // Number of foods eaten by this animal
    pub(crate) satiation: usize,
}

impl Animal {
    pub fn random(rng: &mut dyn RngCore) -> Self {
        let eye = Eye::default();
        let brain = Brain::random(rng, &eye);

        Self::new(eye, brain, rng)
    }

    pub(crate) fn from_chromosome(
        chromosome: ga::Chromosome,
        rng: &mut dyn RngCore,
    ) -> Self {
        let eye = Eye::default();
        let brain = Brain::from_chromosome(chromosome, &eye);

        Self::new(eye, brain, rng)
    }

    pub(crate) fn as_chromosome(&self) -> ga::Chromosome {
        // We evolve only our birds' brains, but technically there's no
        // reason not to simulate e.g. physical properties such as size.
        //
        // If that was to happen, this function could be adjusted to
        // return a longer chromosome that encodes not only the brain,
        // but also, say, birdie's color.

        self.brain.as_chromosome()
    }

    pub fn position(&self) -> na::Point2<f32> {
        self.position
    }
    pub fn rotation(&self) -> na::Rotation2<f32> {
        self.rotation
    }
    pub fn speed(&self) -> f32 {
        self.speed
    }

    fn new(eye: Eye, brain: Brain, rng: &mut dyn RngCore) -> Self {
        Self {
            position: rng.r#gen(),
            rotation: rng.r#gen(),
            speed: 0.002,
            eye,
            brain,
            satiation: 0,
        }
    }


}