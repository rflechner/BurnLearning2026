use std::{io, path::Path as FsPath, path::PathBuf, sync::Arc};

use axum::{
    Json, Router,
    extract::{Path, Query, State, rejection::QueryRejection},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use serde::{Deserialize, Serialize};

use crate::dataset::{self, AnnotatedImage, FaceBox};

pub struct DatasetState {
    entries: Vec<AnnotatedImage>,
    images_root: PathBuf,
}

impl DatasetState {
    pub fn load(annotations: &FsPath, images_root: &FsPath) -> io::Result<Self> {
        // Fail on malformed annotations so IDs always match the file's order.
        let entries = dataset::load_annotations(annotations)?.collect::<io::Result<Vec<_>>>()?;
        let images_root = images_root.canonicalize()?;
        println!("Loaded {} annotated images", entries.len());
        Ok(Self {
            entries,
            images_root,
        })
    }
}

pub fn router(state: DatasetState) -> Router {
    Router::new()
        .route("/api/train", get(list_images))
        .route("/api/train/{id}/image", get(get_image))
        .with_state(Arc::new(state))
}

#[derive(Deserialize)]
#[serde(default)]
struct Pagination {
    page: usize,
    page_size: usize,
}

impl Default for Pagination {
    fn default() -> Self {
        Self {
            page: 1,
            page_size: 20,
        }
    }
}

#[derive(Serialize)]
struct ImageEntry<'a> {
    id: usize,
    image_file_path: &'a str,
    image_url: String,
    faces: &'a [FaceBox],
}

#[derive(Serialize)]
struct Page<'a> {
    page: usize,
    page_size: usize,
    total: usize,
    total_pages: usize,
    items: Vec<ImageEntry<'a>>,
}

#[derive(Serialize)]
struct ErrorBody {
    error: &'static str,
}

struct ApiError(StatusCode, &'static str);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(ErrorBody { error: self.1 })).into_response()
    }
}

async fn list_images(
    State(state): State<Arc<DatasetState>>,
    query: Result<Query<Pagination>, QueryRejection>,
) -> Result<Response, ApiError> {
    let Query(pagination) =
        query.map_err(|_| ApiError(StatusCode::BAD_REQUEST, "Invalid pagination"))?;
    let Pagination { page, page_size } = pagination;
    if page == 0 || !(1..=100).contains(&page_size) {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "page must be >= 1 and page_size between 1 and 100",
        ));
    }
    let start = (page - 1)
        .checked_mul(page_size)
        .ok_or(ApiError(StatusCode::BAD_REQUEST, "Pagination overflow"))?;
    let total = state.entries.len();
    let items = state
        .entries
        .iter()
        .enumerate()
        .skip(start)
        .take(page_size)
        .map(|(id, entry)| ImageEntry {
            id,
            image_file_path: &entry.image_file_path,
            image_url: format!("/api/train/{id}/image"),
            faces: &entry.faces,
        })
        .collect();
    Ok(Json(Page {
        page,
        page_size,
        total,
        total_pages: total.div_ceil(page_size),
        items,
    })
    .into_response())
}

async fn get_image(
    State(state): State<Arc<DatasetState>>,
    Path(id): Path<usize>,
) -> Result<Response, ApiError> {
    let entry = state
        .entries
        .get(id)
        .ok_or(ApiError(StatusCode::NOT_FOUND, "Unknown image"))?;
    let path = tokio::fs::canonicalize(state.images_root.join(&entry.image_file_path))
        .await
        .map_err(image_io_error)?;
    // Even annotations containing '..', absolute paths or symlinks cannot expose external files.
    if !path.starts_with(&state.images_root) {
        return Err(ApiError(StatusCode::NOT_FOUND, "Image not available"));
    }
    let bytes = tokio::fs::read(path).await.map_err(image_io_error)?;
    Ok(([(header::CONTENT_TYPE, "image/jpeg")], bytes).into_response())
}

fn image_io_error(error: io::Error) -> ApiError {
    if error.kind() == io::ErrorKind::NotFound {
        ApiError(StatusCode::NOT_FOUND, "Image not available")
    } else {
        eprintln!("Image read error: {error}");
        ApiError(StatusCode::INTERNAL_SERVER_ERROR, "Unable to read image")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::Request,
    };
    use tower::ServiceExt;

    fn fixture() -> (tempfile::TempDir, Router) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("images");
        std::fs::create_dir_all(root.join("event")).unwrap();
        let mut jpeg = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(2, 2)
            .write_to(&mut jpeg, image::ImageFormat::Jpeg)
            .unwrap();
        std::fs::write(root.join("event/first.jpg"), jpeg.into_inner()).unwrap();
        std::fs::write(dir.path().join("outside.jpg"), b"private").unwrap();
        let annotations = dir.path().join("annotations.txt");
        std::fs::write(&annotations, "event/first.jpg\n1\n1 2 3 4 0 0 0 0 0 0\nevent/missing.jpg\n0\n0 0 0 0 0 0 0 0 0 0\n../outside.jpg\n0\n").unwrap();
        let app = router(DatasetState::load(&annotations, &root).unwrap());
        (dir, app)
    }

    async fn request(app: Router, uri: &str) -> Response {
        app.oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn pagination_and_annotations() {
        let (_dir, app) = fixture();
        let response = request(app.clone(), "/api/train?page=1&page_size=1").await;
        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["total"], 3);
        assert_eq!(body["total_pages"], 3);
        assert_eq!(body["items"][0]["faces"][0]["width"], 3);
        assert_eq!(body["items"][0]["image_url"], "/api/train/0/image");
        let response = request(app.clone(), "/api/train?page=2&page_size=1").await;
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["items"][0]["id"], 1);
        let response = request(app, "/api/train?page=4&page_size=1").await;
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["items"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn rejects_invalid_pagination() {
        let (_dir, app) = fixture();
        for query in [
            "page=0",
            "page_size=0",
            "page_size=101",
            "page=abc",
            "page=-1",
            "page=18446744073709551615&page_size=100",
        ] {
            let response = request(app.clone(), &format!("/api/train?{query}")).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{query}");
        }
    }

    #[tokio::test]
    async fn serves_jpeg_and_handles_missing_or_escaping_images() {
        let (_dir, app) = fixture();
        let response = request(app.clone(), "/api/train/0/image").await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "image/jpeg");
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(image::load_from_memory(&bytes).unwrap().width(), 2);
        for id in [1, 2, 99] {
            assert_eq!(
                request(app.clone(), &format!("/api/train/{id}/image"))
                    .await
                    .status(),
                StatusCode::NOT_FOUND
            );
        }
    }
}
