use anyhow::{Error, Result};
use clap::Parser;
use image::{DynamicImage, GenericImageView};
use std::cmp::Ordering;
use std::cmp::PartialOrd;
use tract_ndarray::{s, ArrayBase, Dim, IxDynImpl, OwnedRepr};
use tract_onnx::prelude::*;

#[derive(Debug, Clone)]
pub struct Bbox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
}

impl Bbox {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32, confidence: f32) -> Bbox {
        Bbox {
            x1,
            y1,
            x2,
            y2,
            confidence,
        }
    }
    pub fn to_xywh(&self) -> (f32, f32, f32, f32) {
        let width = self.x2 - self.x1;
        let height = self.y2 - self.y1;
        let center_x = self.x1 + width / 2.0;
        let center_y = self.y1 + height / 2.0;

        (center_x, center_y, width, height)
    }
    pub fn apply_image_scale(
        &mut self,
        original_image: &DynamicImage,
        x_scale: f32,
        y_scale: f32,
    ) -> Bbox {
        let normalized_x1 = self.x1 / x_scale;
        let normalized_x2 = self.x2 / x_scale;
        let normalized_y1 = self.y1 / y_scale;
        let normalized_y2 = self.y2 / y_scale;

        let cart_x1 = original_image.width() as f32 * normalized_x1;
        let cart_x2 = original_image.width() as f32 * normalized_x2;
        let cart_y1 = original_image.height() as f32 * normalized_y1;
        let cart_y2 = original_image.height() as f32 * normalized_y2;

        Bbox {
            x1: cart_x1,
            y1: cart_y1,
            x2: cart_x2,
            y2: cart_y2,
            confidence: self.confidence,
        }
    }
    pub fn crop_bbox(&self, original_image: &DynamicImage) -> Result<DynamicImage, Error> {
        let bbox_width = (self.x2 - self.x1) as u32;
        let bbox_height = (self.y2 - self.y1) as u32;
        Ok(original_image.to_owned().crop_imm(
            self.x1 as u32,
            self.y1 as u32,
            bbox_width,
            bbox_height,
        ))
    }
}
