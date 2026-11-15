use std::{io, path::Path as FsPath, path::PathBuf, sync::Arc};

use axum::{
    Json, Router,
    extract::{Path, Query, State, rejection::QueryRejection},
    http::{StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::get,
};
use serde::{Deserialize, Serialize};

use crate::dataset::{self, AnnotatedImage, FaceBox};

pub struct DatasetState {
    entries: Vec<AnnotatedImage>,
    images_root: PathBuf,
    inference: std::sync::Mutex<Option<(PathBuf, std::time::SystemTime, crate::model::Model)>>,
}

impl DatasetState {
    pub fn load(annotations: &FsPath, images_root: &FsPath) -> io::Result<Self> {
        // Fail on malformed annotations so IDs always match the file's order.
        let entries = dataset::load_annotations(annotations)?
            .filter(|sample| {
                if let Ok(item) = sample {
                    item.faces.len() <= 2
                } else {
                    false
                }
            })
            .take(50)
            .collect::<io::Result<Vec<_>>>()?;

        let images_root = images_root.canonicalize()?;
        println!("Loaded {} annotated images", entries.len());
        Ok(Self {
            entries,
            images_root,
            inference: std::sync::Mutex::new(None),
        })
    }
}

pub fn router(state: DatasetState) -> Router {
    Router::new()
        .route("/", get(|| async { Html(include_str!("viewer.html")) }))
        .route("/api/train", get(list_images))
        .route("/api/weights", get(list_weights))
        .route("/api/train/{id}/predicted", get(get_predicted_image))
        .route("/api/train/{id}/annotated", get(get_annotated_image))
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
    annotated_image_url: String,
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
            annotated_image_url: format!("/api/train/{id}/annotated"),
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
    let bytes = read_image(&state, id).await?;
    Ok(([(header::CONTENT_TYPE, "image/jpeg")], bytes).into_response())
}

async fn read_image(state: &DatasetState, id: usize) -> Result<Vec<u8>, ApiError> {
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
    tokio::fs::read(path).await.map_err(image_io_error)
}

async fn get_annotated_image(
    State(state): State<Arc<DatasetState>>,
    Path(id): Path<usize>,
) -> Result<Response, ApiError> {
    let bytes = read_image(&state, id).await?;
    // Keep CPU-heavy decoding/drawing/encoding off Tokio's async workers.
    let png = tokio::task::spawn_blocking(move || {
        crate::render::annotated_png(&bytes, &state.entries[id].faces)
    })
    .await
    .map_err(|error| {
        eprintln!("Image rendering task failed: {error}");
        ApiError(StatusCode::INTERNAL_SERVER_ERROR, "Unable to render image")
    })?
    .map_err(|error| {
        eprintln!("Image rendering failed: {error}");
        ApiError(StatusCode::INTERNAL_SERVER_ERROR, "Unable to render image")
    })?;
    Ok(([(header::CONTENT_TYPE, "image/png")], png).into_response())
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
        image::DynamicImage::new_rgb8(12, 12)
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
        assert_eq!(image::load_from_memory(&bytes).unwrap().width(), 12);
        for id in [1, 2, 99] {
            assert_eq!(
                request(app.clone(), &format!("/api/train/{id}/image"))
                    .await
                    .status(),
                StatusCode::NOT_FOUND
            );
        }
    }
    #[tokio::test]
    async fn serves_viewer_and_server_rendered_overlay() {
        let (_dir, app) = fixture();
        let response = request(app.clone(), "/").await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            response.headers()[header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .starts_with("text/html")
        );
        let response = request(app.clone(), "/api/train/0/annotated").await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "image/png");
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let image = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(image.dimensions(), (12, 12));
        assert_eq!(image.get_pixel(1, 2).0, [40, 230, 120]);
        for id in [1, 2, 99] {
            assert_eq!(
                request(app.clone(), &format!("/api/train/{id}/annotated"))
                    .await
                    .status(),
                StatusCode::NOT_FOUND
            );
        }
    }
}

fn weights_root() -> PathBuf {
    std::env::var_os("WEIGHTS_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("runs"))
}

async fn list_weights() -> Result<Json<Vec<String>>, ApiError> {
    let paths = tokio::task::spawn_blocking(|| {
        fn visit(root: &FsPath, dir: &FsPath, result: &mut Vec<String>) -> io::Result<()> {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let kind = entry.file_type()?;
                if kind.is_symlink() {
                    continue;
                }
                if kind.is_dir() {
                    visit(root, &entry.path(), result)?;
                } else if entry.path().extension().is_some_and(|e| e == "bpk") {
                    result.push(
                        entry
                            .path()
                            .strip_prefix(root)
                            .unwrap()
                            .to_string_lossy()
                            .replace('\\', "/"),
                    );
                }
            }
            Ok(())
        }
        let root = weights_root();
        let mut paths = Vec::new();
        if root.exists() {
            visit(&root, &root, &mut paths)?;
        }
        paths.sort();
        Ok::<_, io::Error>(paths)
    })
    .await
    .map_err(|_| {
        ApiError(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Lecture des poids impossible",
        )
    })?
    .map_err(|_| {
        ApiError(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Lecture des poids impossible",
        )
    })?;
    Ok(Json(paths))
}

#[derive(Deserialize)]
struct PredictionQuery {
    weights: String,
    #[serde(default = "default_threshold")]
    threshold: f32,
    #[serde(default = "default_annotations")]
    annotations: bool,
}
fn default_threshold() -> f32 {
    0.5
}
fn default_annotations() -> bool {
    true
}

async fn get_predicted_image(
    State(state): State<Arc<DatasetState>>,
    Path(id): Path<usize>,
    query: Result<Query<PredictionQuery>, QueryRejection>,
) -> Result<Response, ApiError> {
    let Query(query) =
        query.map_err(|_| ApiError(StatusCode::BAD_REQUEST, "Paramètres invalides"))?;
    if !query.threshold.is_finite() || !(0.0..=1.0).contains(&query.threshold) {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "Seuil attendu entre 0 et 1",
        ));
    }
    let relative = FsPath::new(&query.weights);
    if relative
        .components()
        .any(|c| !matches!(c, std::path::Component::Normal(_)))
        || relative.extension().is_none_or(|e| e != "bpk")
    {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "Fichier de poids invalide",
        ));
    }
    let root = tokio::fs::canonicalize(weights_root())
        .await
        .map_err(|_| ApiError(StatusCode::NOT_FOUND, "Dossier de poids absent"))?;
    let path = tokio::fs::canonicalize(root.join(relative))
        .await
        .map_err(|_| ApiError(StatusCode::NOT_FOUND, "Poids introuvables"))?;
    if !path.starts_with(&root) {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "Fichier de poids invalide",
        ));
    }
    let bytes = read_image(&state, id).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, String> {
        let modified = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .map_err(|e| e.to_string())?;
        // Sérialiser l'accès GPU et réutiliser le modèle tant que le fichier est inchangé.
        let mut cache = state
            .inference
            .lock()
            .map_err(|_| "Modèle indisponible : redémarrez le serveur".to_string())?;
        if cache
            .as_ref()
            .is_none_or(|(p, m, _)| p != &path || *m != modified)
        {
            let model = crate::infer::load(&path)?;
            *cache = Some((path, modified, model));
        }
        let detections =
            crate::infer::predict(&cache.as_ref().unwrap().2, &bytes, query.threshold)?;
        let faces = if query.annotations {
            state.entries[id].faces.as_slice()
        } else {
            &[]
        };
        crate::infer::overlay(&bytes, faces, &detections)
    })
    .await
    .map_err(|e| {
        eprintln!("Inférence : {e}");
        ApiError(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Échec de l'inférence ; consultez le terminal",
        )
    })?;
    let png = result.map_err(|e| {
        eprintln!("Inférence : {e}");
        ApiError(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Poids incompatibles ou inférence impossible ; consultez le terminal",
        )
    })?;
    Ok((
        [
            (header::CONTENT_TYPE, "image/png"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        png,
    )
        .into_response())
}
