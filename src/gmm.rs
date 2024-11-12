use nalgebra::{Vector3, Matrix3};
use rand::distributions::{Distribution};
use statrs::distribution::{Uniform, Normal, Continuous};
use rayon::prelude::*;
use std::cmp::Ordering;

pub trait EM<T> {
    fn initialization(&mut self) -> ();
    fn normalization(&mut self) -> ();
    fn expectation(&mut self) -> ();
    fn maximization(&mut self) -> ();
    fn log_likelihood(&mut self) -> ();
    fn run(&mut self) -> ();
    fn predict(&mut self, test_samples: & Vec<T>) -> Vec<usize>;
}


#[allow(dead_code)]
pub struct MVN {
    dim: usize,
    mean: Vector3<f32>,
    cov: Matrix3<f32>,
    det_sqrt: f32,
    inv: Matrix3<f32>,
    cst: f32,
}


pub fn init_mvn( dim: usize, mean: Vector3<f32>, cov: Matrix3<f32>) -> MVN {
        
        let chol = cov.clone().cholesky().unwrap();
        let det_sqrt = chol.l().determinant();
        let cst = 1./((2.* std::f32::consts::PI).powf(dim as f32 / 2.) * chol.l().determinant());

        return MVN {
            dim: dim,
            mean: mean,
            cov: cov,
            det_sqrt: det_sqrt,
            inv: chol.inverse(),
            cst: cst,
        }
    }

impl MVN {
    pub fn pdf(&self, x: &Vector3<f32>) -> f32 { 
        return self.cst*(-0.5*(x-self.mean).transpose()*self.inv*(x-self.mean))[0].exp()
     }
}



pub struct GMM3D<'a> {
    nb_samples: usize,
    nb_components: usize,
    nb_iter: usize,
    epsilon: f32,
    samples: &'a Vec<Vector3<f32>>,
    pub pi: Vec<f32>,
    pub means: Vec<Vector3<f32>>,
    pub covs: Vec<Matrix3<f32>>,
    pub cov_reg: Matrix3<f32>,
    pub mvns: Vec<MVN>,
    pub z: Vec<usize>,
    weights: Vec<f32>,
    gamma_norm: Vec<f32>,
    gamma: Vec<Vec<f32>>,
    pub log_likelihoods: Vec<f32>,
}


pub fn init_gmm3_d<'a>( nb_samples: usize,
    nb_components: usize,
    cov_reg: f32,
    nb_iter:usize,
    epsilon:f32,
    samples: &'a Vec<Vector3<f32>>,
    priors: Vec::<f32>,
    inital_means: Vec::<Vector3<f32>>,
    inital_cov: Vec::<Matrix3<f32>>) -> GMM3D<'a>{
        return GMM3D {
            nb_samples: nb_samples,
            nb_components: nb_components,
            nb_iter: nb_iter,
            epsilon: epsilon,
            samples: &samples,
            pi: priors,
            means: inital_means,
            covs: inital_cov,
            cov_reg: Matrix3::from_diagonal_element(cov_reg),
            mvns: Vec::<MVN>::new(),
            z: Vec::<usize>::new(),
            weights: Vec::<f32>::new(),
            gamma_norm: Vec::<f32>::new(),
            gamma: Vec::<Vec::<f32>>::new(),
            log_likelihoods: Vec::<f32>::new(),
        }
    }


impl<'a> EM<Vector3<f32>> for GMM3D<'a> {

    fn initialization(&mut self) {



        self.weights = vec![0 as f32; self.nb_components];
        self.gamma_norm = vec![0 as f32; self.nb_samples];
        self.gamma = vec![vec![0 as f32; self.nb_components]; self.nb_samples];
        self.z = vec![0 as usize; self.nb_samples];




        self.mvns = self.means
            .iter()
            .zip(self.covs.iter())
            .map(|(&mean, &cov)| init_mvn(3, mean.clone(), cov.clone()))
            .collect::<Vec<_>>();

        // println!("pi {:#?}", self.pi);
        // println!("means {:#?}", self.means);
        // println!("covs {:#?}", self.covs);

    }

    fn normalization(&mut self) {
        // for i in 0..self.nb_samples {
        //     self.gamma_norm[i] = 0 as f32;
        //     for j in 0..self.nb_components {
        //         let mvn = MultivariateNormal::from_mean_and_covariance(&self.means[j], &self.covs[j]).unwrap();
        //         self.gamma_norm[i] += self.pi[j] * mvn.pdf(&(self.samples[i]).transpose())[0];
        //     }
        // }
        
        
        self.gamma_norm = self.samples
            .par_iter()
            .map(|&x| self.mvns.iter()
                .zip(self.pi.iter())
                .map(|(mvn, &p)| p * mvn.pdf(&x))
                .sum()
            ).collect::<Vec<f32>>();
    }

    fn expectation(&mut self) {
        
        // for i in 0..self.nb_samples {
        //     for j in 0..self.nb_components {
        //         let mvn = MultivariateNormal::from_mean_and_covariance(&self.means[j], &self.covs[j]).unwrap();
        //         self.gamma[i][j] = self.pi[j]*mvn.pdf(&(self.samples[i]).transpose())[0]/self.gamma_norm[i];
        //     }
        // }

        self.gamma = self.samples
            .par_iter()
            .zip(self.gamma_norm.par_iter())
            .map(|(&x, cst)| self.mvns.iter()
                .zip(self.pi.iter())
                .map(|(mvn, &p)| p * mvn.pdf(&x)/cst)
                .collect::<Vec<f32>>()
            ).collect::<Vec<Vec<f32>>>();
    }

    fn maximization(&mut self) {
        
        // for i in 0..self.nb_samples {
        //     let mut max_value = std::f32::NEG_INFINITY;
        //     let mut max_idx = 0;
        //     for j in 0..self.nb_components {
        //         if self.gamma[i][j] > max_value {
        //             max_value = self.gamma[i][j];
        //             max_idx = j;
        //         }
        //     }
        //     self.z[i] = max_idx;
        // }

        self.z = self.gamma
            .par_iter()
            .map(|gamma| gamma
                .iter()
                .enumerate()
                .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap()
            )
            .collect::<Vec<usize>>();

        for j in 0..self.nb_components {
            self.weights[j] = 0 as f32;
            for i in 0..self.nb_samples {
                self.weights[j] += self.gamma[i][j];
            }
        }

        self.pi = self.weights
            .iter()
            .map(|x| x / self.nb_samples as f32)
            .collect::<Vec<f32>>();

        for j in 0..self.nb_components {
            // self.pi[j] = self.weights[j]/self.nb_samples as f32;

            self.means[j] = Vector3::from_vec(vec![0.0;3]);
            for i in 0..self.nb_samples {
                self.means[j] += self.gamma[i][j]*self.samples[i];
            }
            self.means[j] /= self.weights[j];
    
    
            self.covs[j] = Matrix3::from_vec(vec![0.0;9]);
            for i in 0..self.nb_samples {
                self.covs[j] += self.gamma[i][j]*(self.samples[i]-self.means[j])*(self.samples[i]-self.means[j]).transpose();
            }
            self.covs[j] /= self.weights[j];
            self.covs[j] += self.cov_reg;

            self.mvns[j] = init_mvn(3, self.means[j].clone(), self.covs[j].clone());
        }


        // self.gaussians = self.means.par_iter()
        //     .zip(self.covs.par_iter())
        //     .map(|(&mean, &cov)| Normal::new(mean, cov).unwrap())
        //     .collect::<Vec<_>>();
    }

    fn log_likelihood(&mut self) {

        // // let mut sum_log = 0 as f32;
        // // for i in 0..self.nb_samples {
        // //     sum_log += self.gamma_norm[i].ln();
        // // }

        let sum_log = self.gamma_norm
            .par_iter()
            .map(|gamma_norm| gamma_norm.ln())
            .sum::<f32>();

        self.log_likelihoods.push(sum_log);
    }

    fn run(&mut self) {
        self.initialization();
        self.normalization();
        self.log_likelihood();
        for tok in 0..self.nb_iter {
            self.expectation();
            self.maximization();
            self.normalization();
            let temp = self.log_likelihoods.last().copied().unwrap();
            self.log_likelihood();
            let error = temp - self.log_likelihoods.last().copied().unwrap();
            let rel_error = 2.*error / (temp + self.log_likelihoods.last().copied().unwrap());
            //println!("Iteration {:4} -- rel_error {:.6} ", tok, rel_error);
            
            if rel_error.abs() < self.epsilon as f32 {
                break;
            }
        }
        // println!("pi {:#?}", self.pi);
        // println!("means {:#?}", self.means);
        // println!("covs {:#?}", self.covs);
    }

    fn predict(&mut self, test_samples: &Vec<Vector3<f32>>) -> Vec<usize> {
        

        // let nb_samples_test = test_samples.len();

        // let mut test_gamma_norm = vec![0.0; nb_samples_test];
        // for i in 0..nb_samples_test {
        //     test_gamma_norm[i] = 0 as f32;
        //     for j in 0..self.nb_components {
        //         let mvn = MultivariateNormal::from_mean_and_covariance(&self.means[j], &self.covs[j]).unwrap();
        //         test_gamma_norm[i] += self.pi[j] * mvn.pdf(&(test_samples[i]).transpose())[0];
        //     }
        // }

        // let mut test_gamma= vec![vec![0.0; self.nb_components]; nb_samples_test];
        // for i in 0..nb_samples_test {
        //     for j in 0..self.nb_components {
        //         let mvn = MultivariateNormal::from_mean_and_covariance(&self.means[j], &self.covs[j]).unwrap();
        //         test_gamma[i][j] = self.pi[j]*mvn.pdf(&(test_samples[i]).transpose())[0]/test_gamma_norm[i];
        //     }
        // }

        let test_gamma_norm = test_samples
            .par_iter()
            .map(|&x| self.mvns.iter()
                .zip(self.pi.iter())
                .map(|(mvn, &p)|
                    p * mvn.pdf(&x)
                )
                .sum()
            ).collect::<Vec<f32>>();
            
        let test_gamma = test_samples
            .par_iter()
            .zip(test_gamma_norm.par_iter())
            .map(|(&x, cst)| self.mvns.iter()
                .zip(self.pi.iter())
                .map(|(mvn, &p)| p * mvn.pdf(&x)/cst)
                .collect::<Vec<f32>>()
            ).collect::<Vec<Vec<f32>>>();

        let test_z = test_gamma
            .par_iter()
            .map(|gamma| gamma.iter()
                .enumerate()
                .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap()
            )
            .collect::<Vec<usize>>();

        return test_z
    }
}