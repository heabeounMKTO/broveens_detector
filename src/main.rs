mod get_face;
mod cv_utils;

use get_face::{get_face, load_model, preprocess_image};
use cv_utils::{get_webcam_stream, mat_to_dynamic_image};
use opencv::prelude::*;
use opencv::core::{Mat, Rect, Scalar, Point};
use opencv::{
    highgui,
    prelude::*,
    videoio::{self, VideoCapture},
};


fn main()  {

    let _load = load_model("models/yolov8n-face_kpss.onnx");
    let window = "test_vdo";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE).unwrap();
    let mut WEBCAM = get_webcam_stream(0).unwrap();
    loop {
        let mut frame = Mat::default();
        WEBCAM.read(&mut frame).unwrap();
        if frame.size().unwrap().width > 0 {
            let mat2dyn = mat_to_dynamic_image(&frame);
            let face = get_face(&_load,&mat2dyn, 0.8, 0.6).unwrap();
            let _xywh;
            if face.len() > 0 {
                _xywh = face[0].clone();
                let top_left = Point::new(_xywh.x1 as i32, _xywh.y1 as i32);
                let color = Scalar::new(0.0, 255.0, 0.0, 0.0); // Green color
                opencv::imgproc::rectangle(&mut frame, Rect::new(top_left.x, top_left.y, (_xywh.x2 - _xywh.x1) as i32, (_xywh.y2 - _xywh.y1) as i32), color, 2, opencv::imgproc::LINE_8, 0).unwrap();
                highgui::imshow(window, &frame).unwrap();
            }
            else {
                highgui::imshow(window, &frame).unwrap();
            }
        }
        let key = highgui::wait_key(10).unwrap();
        if key > 0 && key != 255 {
            break;
        }
    } 
}
