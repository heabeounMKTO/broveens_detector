use crate::bbox::Bbox;
use anyhow::{Error, Result};
use clap::Parser;
use crate::bbox::{non_maximum_suppression};
use image::{DynamicImage, GenericImageView};
use std::cmp::Ordering;
use std::cmp::PartialOrd;
use tract_ndarray::{s, ArrayBase, Dim, IxDynImpl, OwnedRepr};
use tract_onnx::prelude::*;
/// loads model, just panics if anything goes wrong
/// THATS BY DESIGN
pub fn load_model(
    model_path: &str,
) -> SimplePlan<TypedFact, Box<dyn TypedOp>, tract_onnx::prelude::Graph<TypedFact, Box<dyn TypedOp>>>
{
    let model = tract_onnx::onnx()
        .model_for_path(model_path)
        .unwrap()
        .with_input_fact(0, f32::fact([1, 3, 320, 320]).into())
        .unwrap()
        .into_optimized()
        .unwrap()
        .into_runnable()
        .unwrap();
    model
}

/// add black bars padding to image
pub fn preprocess_image(raw_image: &DynamicImage) -> Result<Tensor> {
    let width = raw_image.width();
    let height = raw_image.height();
    let scale = 320.0 / width.max(height) as f32;
    let new_width = (width as f32 * scale) as u32;
    let new_height = (height as f32 * scale) as u32;
    let resized = image::imageops::resize(
        &raw_image.to_rgb8(),
        new_width,
        new_height,
        image::imageops::FilterType::Triangle,
    );
    let mut padded = image::RgbImage::new(320, 320);
    image::imageops::replace(
        &mut padded,
        &resized,
        (320 - new_width as i64) / 2,
        (320 - new_height as i64) / 2,
    );
    let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 320, 320), |(_, c, y, x)| {
        padded.get_pixel(x as u32, y as u32)[c] as f32 / 255.0
    })
    .into();
    Ok(image)
}

fn process_results(
    input_results: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
    input_image: &DynamicImage,
    confidence_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<Bbox>, Error> {
    let mut results_vec: Vec<Bbox> = vec![];
    for i in 0..input_results.len_of(tract_ndarray::Axis(0)) {
        let row = input_results.slice(s![i, .., ..]);
        let confidence = row[[4, 0]];

        if confidence >= confidence_threshold {
            let x = row[[0, 0]];
            let y = row[[1, 0]];
            let w = row[[2, 0]];
            let h = row[[3, 0]];
            let x1 = x - w / 2.0;
            let y1 = y - h / 2.0;
            let x2 = x + w / 2.0;
            let y2 = y + h / 2.0;
            let bbox =
            Bbox::new(x1, y1, x2, y2, confidence).apply_image_scale(&input_image, 320.0, 320.0);
            results_vec.push(bbox);
        }
    }
    Ok(results_vec)
}

pub fn get_face(
    loaded_model: &SimplePlan<
        TypedFact,
        Box<dyn TypedOp>,
        tract_onnx::prelude::Graph<TypedFact, Box<dyn TypedOp>>,
    >,
    input_image: &DynamicImage,
    confidence_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<Bbox>, Error> {
    let preproc = preprocess_image(input_image)?;
    let forward = loaded_model.run(tvec![preproc.to_owned().into()])?;
    let results = forward[0].to_array_view::<f32>()?.view().t().into_owned();
    let _final = process_results(results, input_image, confidence_threshold, iou_threshold)?;
    // do a NMS pass on final bboxes that way we dont need to
    // sort 1000+++++++ boxes for no reason..
    Ok(non_maximum_suppression(_final, iou_threshold))
}

