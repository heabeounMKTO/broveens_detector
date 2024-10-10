use image::{DynamicImage, ImageBuffer, Rgb};
use opencv::core::{self,Mat};
use opencv::prelude::*;
use opencv::{
    highgui,
    prelude::*,
    videoio::{self, VideoCapture},
};
use anyhow::{Error, Result};


pub fn mat_to_dynamic_image(mat: &Mat) -> DynamicImage {
    let rows = mat.rows();
    let cols = mat.cols();
    let channels = mat.channels();

    assert!(channels == 3, "Only 3-channel (RGB) images are supported");
    let mut img_buf: Vec<u8> = vec![0; (rows * cols * channels) as usize];
    let mat_slice = mat.data_bytes().unwrap();
    img_buf.copy_from_slice(mat_slice);
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(cols as u32, rows as u32, img_buf).unwrap();
    DynamicImage::ImageRgb8(img)
}


pub fn get_webcam_stream(index: i32) -> Result<VideoCapture, Error> {
    let mut cam = videoio::VideoCapture::new(index, videoio::CAP_ANY)?;
    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("cannot open def cam!");
    }
    Ok(cam)
}

