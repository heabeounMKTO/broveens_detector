mod bbox;
/// notes:
/// yea bro i know we can `make it proper` and all,
/// BUT this is a demo , we dont need to make a
/// "abstract model loader" struct and enums comeon now
///
mod cv_utils;

#[cfg(not(feature = "use_ort"))]
mod get_face;

#[cfg(feature = "use_ort")]
mod get_face_onnx;

use clap::Parser;
use cv_utils::{get_webcam_stream, mat_to_dynamic_image};

#[cfg(not(feature = "use_ort"))]
use get_face::{get_face, load_model, preprocess_image};

#[cfg(feature = "use_ort")]
use get_face_onnx::{get_face, load_model, preprocess_image};

use opencv::core::{Mat, Point, Rect, Scalar};
use opencv::prelude::*;
use opencv::{
    highgui,
    prelude::*,
    videoio::{self, VideoCapture},
};
use std::time::Instant;

fn main() {
    // NO ENUM NEEDED
    // FUCK THEM ENUM KIDS
    let _load = load_model("models/yolov8n_face.onnx");
    let window = "broveen_detect";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE).unwrap();
    let mut WEBCAM = get_webcam_stream(0).unwrap();
    let position = Point::new(10, 30);
    let font = opencv::imgproc::FONT_HERSHEY_SIMPLEX;
    let font_scale = 1.0;
    let color = Scalar::new(0.0, 255.0, 0.0, 0.0);
    let thickness = 2;
    let line_type = 8;
    let bottom_left_origin = false;

    loop {
        let mut frame = Mat::default();
        WEBCAM.read(&mut frame).unwrap();
        if frame.size().unwrap().width > 0 {
            let mat2dyn = mat_to_dynamic_image(&frame);

            let start_inf = Instant::now();
            let face = get_face(&_load, &mat2dyn, 0.8, 0.6).unwrap();
            let inf_time = format!("inference time: {:?}ms", start_inf.elapsed());

            let _xywh;
            if face.len() > 0 {
                _xywh = face[0].clone();
                let top_left = Point::new(_xywh.x1 as i32, _xywh.y1 as i32);
                let color = Scalar::new(0.0, 255.0, 0.0, 0.0); // Green color
                opencv::imgproc::rectangle(
                    &mut frame,
                    Rect::new(
                        top_left.x,
                        top_left.y,
                        (_xywh.x2 - _xywh.x1) as i32,
                        (_xywh.y2 - _xywh.y1) as i32,
                    ),
                    color,
                    2,
                    opencv::imgproc::LINE_8,
                    0,
                )
                .unwrap();

                opencv::imgproc::put_text(
                    &mut frame,
                    &inf_time,
                    position,
                    font,
                    font_scale,
                    color,
                    thickness,
                    line_type,
                    bottom_left_origin,
                )
                .unwrap();
                highgui::imshow(window, &frame).unwrap();
            } else {
                highgui::imshow(window, &frame).unwrap();
            }
        }
        let key = highgui::wait_key(10).unwrap();
        if key > 0 && key != 255 {
            break;
        }
    }
}
