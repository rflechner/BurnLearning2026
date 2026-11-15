//! Dataset overlays, independent of HTTP and filesystem access.
use image::{Rgb, RgbImage};

use crate::dataset::FaceBox;

/// Draw face outlines in image coordinates. Invalid annotations are amber;
/// others are green. Empty boxes are ignored and oversized boxes are clipped.
/// The original image can be retained by passing a clone to this function.
pub fn draw_face_boxes(image: &mut RgbImage, faces: &[FaceBox]) {
    let (width, height) = image.dimensions();
    for face in faces {
        if face.is_degenerate() || face.x >= width || face.y >= height {
            continue;
        }
        let right = face.x.saturating_add(face.width - 1).min(width - 1);
        let bottom = face.y.saturating_add(face.height - 1).min(height - 1);
        let color = if face.invalid != 0 {
            Rgb([255, 180, 40])
        } else {
            Rgb([40, 230, 120])
        };
        for thickness in 0..2 {
            let left = face.x + thickness;
            let top = face.y + thickness;
            let Some(right) = right.checked_sub(thickness) else {
                break;
            };
            let Some(bottom) = bottom.checked_sub(thickness) else {
                break;
            };
            if left > right || top > bottom {
                break;
            }
            for x in left..=right {
                image.put_pixel(x, top, color);
                image.put_pixel(x, bottom, color);
            }
            for y in top..=bottom {
                image.put_pixel(left, y, color);
                image.put_pixel(right, y, color);
            }
        }
    }
}

/// Decode an image, overlay its annotations, and encode a lossless PNG.
pub fn annotated_png(bytes: &[u8], faces: &[FaceBox]) -> image::ImageResult<Vec<u8>> {
    let mut image = image::load_from_memory(bytes)?.to_rgb8();
    draw_face_boxes(&mut image, faces);
    let mut output = std::io::Cursor::new(Vec::new());
    image.write_to(&mut output, image::ImageFormat::Png)?;
    Ok(output.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(x: u32, y: u32, width: u32, height: u32, invalid: u32) -> FaceBox {
        FaceBox {
            x,
            y,
            width,
            height,
            invalid,
            blur: 0,
            expression: 0,
            illumination: 0,
            occlusion: 0,
            pose: 0,
        }
    }

    #[test]
    fn draws_outlines_without_filling_the_face() {
        let mut image = RgbImage::new(12, 12);
        draw_face_boxes(&mut image, &[face(2, 2, 8, 8, 0)]);
        assert_eq!(image.get_pixel(2, 2).0, [40, 230, 120]);
        assert_eq!(image.get_pixel(9, 9).0, [40, 230, 120]);
        assert_eq!(image.get_pixel(5, 5).0, [0, 0, 0]);
        assert_eq!(image.get_pixel(10, 10).0, [0, 0, 0]);
    }

    #[test]
    fn handles_empty_offscreen_and_overflowing_boxes() {
        let mut image = RgbImage::new(8, 8);
        draw_face_boxes(
            &mut image,
            &[face(0, 0, 0, 8, 0), face(u32::MAX, 0, 8, 8, 0)],
        );
        assert!(image.pixels().all(|p| p.0 == [0, 0, 0]));
        draw_face_boxes(&mut image, &[face(6, 6, u32::MAX, u32::MAX, 1)]);
        assert_eq!(image.get_pixel(7, 7).0, [255, 180, 40]);
        draw_face_boxes(&mut RgbImage::new(0, 0), &[face(0, 0, 1, 1, 0)]);
    }
}
