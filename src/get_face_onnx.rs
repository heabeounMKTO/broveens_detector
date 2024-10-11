use crate::bbox::Bbox;
use anyhow::{Error, Result};
use image::imageops;
use image::{DynamicImage, GenericImageView};
#[cfg(feature="use_ort")]
use ndarray::{s, Array, ArrayBase, Axis, Dim, IxDyn, OwnedRepr};
use ort::{Session, CPUExecutionProvider, GraphOptimizationLevel, inputs};
use crate::bbox::non_maximum_suppression;

pub fn load_model(model_path: &str) -> Session {
    let model: Session = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .unwrap()
        .commit_from_file(model_path)
        .unwrap();
    model
}
fn scale_wh(w0: f32, h0: f32, w1: f32, h1: f32) -> (f32, f32, f32) {
    let r = (w1 / w0).min(h1 / h0);
    (r, (w0 * r).round(), (h0 * r).round())
}

/// add black bars padding to image (i forgot what it originally does so im just saying for
/// reference),
/// ONNXRUNTIME NALGEBRA EDITION NOW
pub fn preprocess_image(
    image_source: &DynamicImage,
) -> Result<ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>, Error> {
    let mut preproc: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> = Array::ones((1, 3, 320, 320));

    // TODO: refactor to funciton arg
    let (_, w_new, h_new) = scale_wh(
        image_source.width() as f32,
        image_source.height() as f32,
        320.0,
        320.0,
    );
    let img = image_source.resize_exact(w_new as u32, h_new as u32, imageops::FilterType::Triangle);
    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2 .0;
        preproc[[0, 0, y, x]] = r as f32 / 255.0;
        preproc[[0, 1, y, x]] = g as f32 / 255.0;
        preproc[[0, 2, y, x]] = b as f32 / 255.0;
    }
    Ok(preproc)
}

pub fn get_face (
    load_model: &Session,
    input_image: &DynamicImage,
    confidence_threshold: f32,
    iou_threshold: f32
) -> Result<Vec<Bbox>, Error> {
    let preprocess_image = preprocess_image(input_image)?;
    let inference = load_model.run(inputs!["images" => preprocess_image.view()]?)?;
    let _raw_output = inference["output0"]
        .try_extract_tensor::<f32>()?
        .view()
        .t()
        .into_owned();
    let (_, w_new, h_new) = scale_wh(
        input_image.width() as f32,
        input_image.height() as f32,
        320.0,
        320.0,
    );
    let mut bbox_vec: Vec<Bbox> = vec![];
    for i in 0.._raw_output.len_of(Axis(0)) {
        let row = _raw_output.slice(s![i, .., ..]);
        let confidence = row[[4, 0]];
        if &confidence >= &confidence_threshold {
            let x = row[[0, 0]];
            let y = row[[1, 0]];
            let w = row[[2, 0]];
            let h = row[[3, 0]];

            let x1 = x - w / 2.0;
            let y1 = y - h / 2.0;
            let x2 = x + w / 2.0;
            let y2 = y + h / 2.0;
            let bbox = Bbox::new(x1, y1, x2, y2, confidence).apply_image_scale(
                &input_image,
                w_new,
                h_new,
            );
            bbox_vec.push(bbox);
        }
    }
    Ok(non_maximum_suppression(bbox_vec, iou_threshold))
}



