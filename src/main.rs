use std::path::Path;
use gmm_based_image_segmentation::gmm::{init_gmm3_d, EM, GMM3D};
use image::Pixel;
use ndarray::{ Array2, Axis};
// Import the linfa prelude and KMeans algorithm
// We'll build our dataset on our own using ndarray and rand
use ndarray::prelude::*;
use ndarray_stats::CorrelationExt;
use num_traits::cast::ToPrimitive;
use nalgebra::*;

// Import the plotters crate to create the scatter plot
use clustering::*;

fn vec_to_ndarray<T>(vec: Vec<Vec<T>>) -> Array2<f32> 
where
    T: ToPrimitive + Copy,
    {
    // Determine the dimensions
    let height = vec.len();
    let width = vec[0].len();

    // Flatten the 2D Vec<Vec<u8>> into a 1D Vec<u8>
    let flattened: Vec<f32> = vec.into_iter().flatten()
        .map(|x| x.to_f32().unwrap_or(0.0) )
        .collect();

    // Create a 2D array with shape (height, width)
    Array2::from_shape_vec((height, width), flattened).expect("Shape should match")

    // Add an extra dimension for channels, making it 3D (height, width, 1)
}

fn array2_to_matrix3(array: Array2<f32>) -> Matrix3<f32> {
    // Ensure the Array2 is 3x3
    if array.nrows() != 3 || array.ncols() != 3 {
        panic!("The input array must be 3x3.");
    }

    // Populate the Matrix3 from the Array2 elements
    Matrix3::new(
        array[(0, 0)], array[(0, 1)], array[(0, 2)],
        array[(1, 0)], array[(1, 1)], array[(1, 2)],
        array[(2, 0)], array[(2, 1)], array[(2, 2)],
    )
}

fn main() {
    let from = "/home/bh/Documents/gmm_based_image_segmentation/data/deer.jpg";
    let mut im: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> = image::open(Path::new(&from)).unwrap().into_rgb8();
    let width = im.width();
    let height= im.height();
    // Iterate over the coordinates and pixels of the image
    let mut channels:Vec<Vec<u8>> = im.enumerate_pixels_mut().map(|(x,y,rgb)| vec![rgb.channels()[0],rgb.channels()[1],rgb.channels()[2]]).collect();
    

    let mut data = vec_to_ndarray(channels);

    data -= &data.mean_axis(Axis(0)).unwrap();

    data /= &data.std_axis(Axis(0), 0.);

    println!("New mean:\n{:8.4}", data.mean_axis(Axis(0)).unwrap());
    println!("New std: \n{:8.4}", data.std_axis(Axis(0), 0.));
    //let std_r= array_2d.var_axis(Axis(0),1.0);

    // Print the shape and array to verify
    println!("Shape: {:?}", data.dim());

    let n_samples    = data.dim().0; // # of samples in the example
    let n_dimensions =    data.dim().1; // # of dimensions in each sample
    let k            =      3; // # of clusters in the result
    let max_iter     =    100; // max number of iterations before the clustering forcefully stops
    let rows: Vec<Vec<f32>> = data.axis_iter(ndarray::Axis(0))
        .map(|row| row.to_vec())
        .collect();

    // actually perform the clustering
    let clustering = kmeans(k, &rows, max_iter);
    let centroids: Vec<Vec<f64>> = clustering.centroids.into_iter().map(|e| e.0).collect();
    println!("membership: {:?}", clustering.membership.len());
   // println!("centroids : {:?}", clustering.centroids);
    println!("centroids : {:?}", centroids);

    let imgx = width as u32 ;
    let imgy = height as u32;
    
    let mut initial_priors: Vec<f32> = vec![];
    let mut initial_covs: Vec<Array2<f32>> = vec![];
    for i in 0..k {
        let selected_rows: Vec<Vec<f32>> = clustering.membership.iter()
            .enumerate()
            .filter_map(|(j, &label)| {
                if label == i {
                    Some(data.row(j).to_vec().to_owned())
                } else {
                    None
                }
            })
            .collect();
            let selected_data = vec_to_ndarray(selected_rows);
            println!("{:?}",selected_data.shape());
            initial_covs.push(selected_data.t().cov( 1.).unwrap());
            initial_priors.push(selected_data.shape()[0] as f32 / clustering.membership.len() as f32);
    // Stack the rows into a new Array2 and transpose it

    }
    println!("cov{:?}",initial_covs);
    println!("prior{:?}",initial_priors);

    let samples_gmm: Vec<Vector3<f32>> = data.axis_iter(ndarray::Axis(0))
        .map(|row| Vector3::from_vec(row.to_vec()))
        .collect();

    let mut means_gmm: Vec<Vector3<f32>> = centroids.into_iter().map(|mean|Vector3::from_vec(
          mean.into_iter().map(|x| x as f32).collect()
        )).collect();

    println!("centro{:?}",means_gmm);
    let mut cov_gmm: Vec<Matrix3<f32>>= vec![];
    for matrix in initial_covs {
        cov_gmm.push(array2_to_matrix3(matrix));
    }
    let mut model = init_gmm3_d(n_samples, 3, 0.001, 10000, 1e-9, &samples_gmm, initial_priors, means_gmm, cov_gmm);
    model.run();
    let predict = model.predict(&samples_gmm);
    // Create a new ImgBuf with width: imgx and height: imgy
    let mut imgbuf = image::ImageBuffer::new(imgx, imgy); 
    
    // Iterate over the coordinates and pixels of the image
    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let index = (x+y*width) as usize;
        match clustering.membership[index] {
            0 => {
                let r =100.0 as u8; let b=10.0 as u8; let g= 0.0 as u8;
                *pixel = image::Rgb([r, g, b]);

            }
            1 => {
                let r =0.0 as u8; let b=10. as u8; let g= 40.0 as u8;
                *pixel = image::Rgb([r, g, b]);

            }
            _ => {
                let r =0.0 as u8; let b=0.0 as u8; let g= 90.0 as u8;
                *pixel = image::Rgb([r, g, b]);

            }

        }

    }
    

    
    // Save the image as “fractal.png”, the format is deduced from the path
    imgbuf.save("result_kmeans.png").unwrap();

    let mut imgbuf = image::ImageBuffer::new(imgx, imgy); 
    
    // Iterate over the coordinates and pixels of the image
    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let index = (x+y*width) as usize;
        match predict[index] {
            0 => {
                let r =100.0 as u8; let b=10.0 as u8; let g= 0.0 as u8;
                *pixel = image::Rgb([r, g, b]);

            }
            1 => {
                let r =0.0 as u8; let b=10. as u8; let g= 40.0 as u8;
                *pixel = image::Rgb([r, g, b]);

            }
            _ => {
                let r =0.0 as u8; let b=0.0 as u8; let g= 90.0 as u8;
                *pixel = image::Rgb([r, g, b]);

            }

        }

    }
    

    
    // Save the image as “fractal.png”, the format is deduced from the path
    imgbuf.save("result_gmm.png").unwrap();
    
    // Save the image as “fractal.png”, the format is deduced from the path
   // imgbuf.save("result.png").unwrap();
}
