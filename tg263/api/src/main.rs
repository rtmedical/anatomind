use axum::{
    extract::{Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use futures_util::{Stream, StreamExt};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    env,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::net::TcpListener;
use tracing::info;
use uuid::Uuid;

#[derive(Debug, Clone)]
struct AppConfig {
    provider: String,
    vllm_url: String,
    tgi_url: String,
    default_chat_model: String,
    tg263_csv: String,
    abstain_sim_threshold: f64,
    max_labels: usize,
}

#[derive(Debug, Clone)]
struct TG263Data {
    labels: Vec<String>,
    normalized_to_original: HashMap<String, String>,
}

type AppState = Arc<(AppConfig, TG263Data)>;

static NORMALIZE_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"[^a-z0-9\s]+").unwrap());
static WHITESPACE_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());

fn normalize_text(text: &str) -> String {
    let lower = text.to_lowercase();
    let no_accents = deunicode::deunicode(&lower);
    let no_special = NORMALIZE_REGEX.replace_all(&no_accents, " ");
    WHITESPACE_REGEX.replace_all(no_special.trim(), " ").to_string()
}

fn load_tg263_data(csv_path: &str) -> anyhow::Result<TG263Data> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b';')
        .from_path(csv_path)?;
    let mut labels = Vec::new();
    let mut normalized_to_original = HashMap::new();

    for result in reader.records() {
        let record = result?;
        // TG263-Primary Name está na coluna 4 (índice 4)
        if let Some(label) = record.get(4) {
            if !label.trim().is_empty() {
                let normalized = normalize_text(label);
                labels.push(normalized.clone());
                normalized_to_original.insert(normalized, label.to_string());
            }
        }
    }

    info!("Loaded {} TG-263 labels from {}", labels.len(), csv_path);
    Ok(TG263Data {
        labels,
        normalized_to_original,
    })
}

#[derive(Deserialize)]
struct ProviderQuery {
    provider: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Deserialize, Serialize)]
struct OpenAIChatRequest {
    model: Option<String>,
    messages: Vec<OpenAIMessage>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<u32>,
    stream: Option<bool>,
    provider: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct OpenAICompletionsRequest {
    model: Option<String>,
    prompt: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<u32>,
    stream: Option<bool>,
    provider: Option<String>,
}

#[derive(Serialize)]
struct OpenAIModel {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

#[derive(Serialize)]
struct OpenAIModelsResponse {
    object: String,
    data: Vec<OpenAIModel>,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    meta: HealthMeta,
}

#[derive(Serialize)]
struct HealthMeta {
    served_model: String,
    default_provider: String,
    labels: usize,
}

#[derive(Deserialize)]
struct ClassifyRequest {
    input: String,
    top_k: Option<usize>,
    provider: Option<String>,
}

#[derive(Serialize)]
struct ClassifyResponse {
    status: String,
    mapped_label: Option<String>,
    topk: Vec<ClassifyResult>,
    raw_output: String,
    provider: String,
}

#[derive(Serialize)]
struct ClassifyResult {
    label: String,
    score: f64,
}

#[derive(Serialize)]
struct TGIGenerateRequest {
    inputs: String,
    parameters: TGIParameters,
    stream: Option<bool>,
}

#[derive(Serialize)]
struct TGIParameters {
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_new_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct TGIResponse {
    generated_text: String,
}

#[derive(Deserialize)]
struct TGIStreamResponse {
    token: TGIToken,
}

#[derive(Deserialize)]
struct TGIToken {
    text: String,
}

async fn healthz(State(state): State<AppState>) -> Json<HealthResponse> {
    let (config, tg263_data) = state.as_ref();
    Json(HealthResponse {
        status: "ok".to_string(),
        meta: HealthMeta {
            served_model: config.default_chat_model.clone(),
            default_provider: config.provider.clone(),
            labels: tg263_data.labels.len(),
        },
    })
}

async fn models(State(state): State<AppState>) -> Json<OpenAIModelsResponse> {
    let (config, _) = state.as_ref();
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    Json(OpenAIModelsResponse {
        object: "list".to_string(),
        data: vec![OpenAIModel {
            id: config.default_chat_model.clone(),
            object: "model".to_string(),
            created,
            owned_by: "anatomind".to_string(),
        }],
    })
}

fn get_provider(config: &AppConfig, query: &Option<String>, body_provider: &Option<String>) -> String {
    body_provider
        .as_ref()
        .or(query.as_ref())
        .unwrap_or(&config.provider)
        .clone()
}

async fn chat_completions(
    Query(query): Query<ProviderQuery>,
    State(state): State<AppState>,
    Json(mut request): Json<OpenAIChatRequest>,
) -> Result<Response, StatusCode> {
    let (config, _) = state.as_ref();
    let provider = get_provider(config, &query.provider, &request.provider);
    
    request.model = Some(config.default_chat_model.clone());
    
    if request.stream.unwrap_or(false) {
        handle_streaming_chat(&provider, config, request).await
    } else {
        handle_non_streaming_chat(&provider, config, request).await
    }
}

async fn completions(
    Query(query): Query<ProviderQuery>,
    State(state): State<AppState>,
    Json(mut request): Json<OpenAICompletionsRequest>,
) -> Result<Response, StatusCode> {
    let (config, _) = state.as_ref();
    let provider = get_provider(config, &query.provider, &request.provider);
    
    request.model = Some(config.default_chat_model.clone());
    
    if request.stream.unwrap_or(false) {
        handle_streaming_completions(&provider, config, request).await
    } else {
        handle_non_streaming_completions(&provider, config, request).await
    }
}

async fn handle_streaming_chat(
    provider: &str,
    config: &AppConfig,
    request: OpenAIChatRequest,
) -> Result<Response, StatusCode> {
    match provider {
        "vllm" => {
            let client = reqwest::Client::new();
            let url = format!("{}/v1/chat/completions", config.vllm_url);
            
            let response = client
                .post(&url)
                .json(&request)
                .send()
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                
            if !response.status().is_success() {
                return Err(StatusCode::BAD_GATEWAY);
            }
            
            let stream = response.bytes_stream();
            let sse_stream = async_stream::stream! {
                let mut stream = std::pin::pin!(stream);
                while let Some(chunk) = stream.next().await {
                    if let Ok(bytes) = chunk {
                        if let Ok(text) = String::from_utf8(bytes.to_vec()) {
                            yield Ok::<_, axum::Error>(axum::response::sse::Event::default().data(text));
                        }
                    }
                }
            };
            
            Ok(Sse::new(sse_stream).into_response())
        }
        "tgi" => {
            let stream = generate_tgi_chat_stream(config, request).await?;
            Ok(Sse::new(stream).into_response())
        }
        _ => Err(StatusCode::BAD_REQUEST),
    }
}

async fn handle_streaming_completions(
    provider: &str,
    config: &AppConfig,
    request: OpenAICompletionsRequest,
) -> Result<Response, StatusCode> {
    match provider {
        "vllm" => {
            let client = reqwest::Client::new();
            let url = format!("{}/v1/completions", config.vllm_url);
            
            let response = client
                .post(&url)
                .json(&request)
                .send()
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                
            if !response.status().is_success() {
                return Err(StatusCode::BAD_GATEWAY);
            }
            
            let stream = response.bytes_stream();
            let sse_stream = async_stream::stream! {
                let mut stream = std::pin::pin!(stream);
                while let Some(chunk) = stream.next().await {
                    if let Ok(bytes) = chunk {
                        if let Ok(text) = String::from_utf8(bytes.to_vec()) {
                            yield Ok::<_, axum::Error>(axum::response::sse::Event::default().data(text));
                        }
                    }
                }
            };
            
            Ok(Sse::new(sse_stream).into_response())
        }
        "tgi" => {
            let stream = generate_tgi_completions_stream(config, request).await?;
            Ok(Sse::new(stream).into_response())
        }
        _ => Err(StatusCode::BAD_REQUEST),
    }
}

async fn handle_non_streaming_chat(
    provider: &str,
    config: &AppConfig,
    request: OpenAIChatRequest,
) -> Result<Response, StatusCode> {
    let client = reqwest::Client::new();
    
    let response = match provider {
        "vllm" => {
            let url = format!("{}/v1/chat/completions", config.vllm_url);
            client.post(&url).json(&request).send().await
        }
        "tgi" => {
            let prompt = format_messages_as_prompt(&request.messages);
            let tgi_request = TGIGenerateRequest {
                inputs: prompt,
                parameters: TGIParameters {
                    temperature: request.temperature,
                    top_p: request.top_p,
                    max_new_tokens: request.max_tokens,
                },
                stream: Some(false),
            };
            let url = format!("{}/generate", config.tgi_url);
            client.post(&url).json(&tgi_request).send().await
        }
        _ => return Err(StatusCode::BAD_REQUEST),
    }
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if !response.status().is_success() {
        return Err(StatusCode::BAD_GATEWAY);
    }

    let body = response.bytes().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(Response::builder()
        .header(header::CONTENT_TYPE, "application/json")
        .body(body.into())
        .unwrap())
}

async fn handle_non_streaming_completions(
    provider: &str,
    config: &AppConfig,
    request: OpenAICompletionsRequest,
) -> Result<Response, StatusCode> {
    let client = reqwest::Client::new();
    
    let response = match provider {
        "vllm" => {
            let url = format!("{}/v1/completions", config.vllm_url);
            client.post(&url).json(&request).send().await
        }
        "tgi" => {
            let tgi_request = TGIGenerateRequest {
                inputs: request.prompt,
                parameters: TGIParameters {
                    temperature: request.temperature,
                    top_p: request.top_p,
                    max_new_tokens: request.max_tokens,
                },
                stream: Some(false),
            };
            let url = format!("{}/generate", config.tgi_url);
            client.post(&url).json(&tgi_request).send().await
        }
        _ => return Err(StatusCode::BAD_REQUEST),
    }
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if !response.status().is_success() {
        return Err(StatusCode::BAD_GATEWAY);
    }

    let body = response.bytes().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(Response::builder()
        .header(header::CONTENT_TYPE, "application/json")
        .body(body.into())
        .unwrap())
}

async fn generate_tgi_chat_stream(
    config: &AppConfig,
    request: OpenAIChatRequest,
) -> Result<impl Stream<Item = Result<axum::response::sse::Event, axum::Error>>, StatusCode> {
    let client = reqwest::Client::new();
    let prompt = format_messages_as_prompt(&request.messages);
    
    let tgi_request = TGIGenerateRequest {
        inputs: prompt,
        parameters: TGIParameters {
            temperature: request.temperature,
            top_p: request.top_p,
            max_new_tokens: request.max_tokens,
        },
        stream: Some(true),
    };
    
    let url = format!("{}/generate_stream", config.tgi_url);
    let response = client
        .post(&url)
        .json(&tgi_request)
        .send()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        
    if !response.status().is_success() {
        return Err(StatusCode::BAD_GATEWAY);
    }
    
    let stream = response.bytes_stream();
    let request_id = Uuid::new_v4().to_string();
    let model = request.model.unwrap_or_default();
    
    let sse_stream = async_stream::stream! {
        let mut stream = std::pin::pin!(stream);
        while let Some(chunk) = stream.next().await {
            if let Ok(bytes) = chunk {
                if let Ok(text) = String::from_utf8(bytes.to_vec()) {
                    for line in text.lines() {
                        if let Some(data) = line.strip_prefix("data: ") {
                            if let Ok(tgi_response) = serde_json::from_str::<TGIStreamResponse>(data) {
                                let openai_chunk = serde_json::json!({
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {
                                            "content": tgi_response.token.text
                                        }
                                    }]
                                });
                                yield Ok(axum::response::sse::Event::default().data(openai_chunk.to_string()));
                            }
                        }
                    }
                }
            }
        }
        yield Ok(axum::response::sse::Event::default().data("[DONE]"));
    };
    
    Ok(sse_stream)
}

async fn generate_tgi_completions_stream(
    config: &AppConfig,
    request: OpenAICompletionsRequest,
) -> Result<impl Stream<Item = Result<axum::response::sse::Event, axum::Error>>, StatusCode> {
    let client = reqwest::Client::new();
    
    let tgi_request = TGIGenerateRequest {
        inputs: request.prompt,
        parameters: TGIParameters {
            temperature: request.temperature,
            top_p: request.top_p,
            max_new_tokens: request.max_tokens,
        },
        stream: Some(true),
    };
    
    let url = format!("{}/generate_stream", config.tgi_url);
    let response = client
        .post(&url)
        .json(&tgi_request)
        .send()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        
    if !response.status().is_success() {
        return Err(StatusCode::BAD_GATEWAY);
    }
    
    let stream = response.bytes_stream();
    let request_id = Uuid::new_v4().to_string();
    let model = request.model.unwrap_or_default();
    
    let sse_stream = async_stream::stream! {
        let mut stream = std::pin::pin!(stream);
        while let Some(chunk) = stream.next().await {
            if let Ok(bytes) = chunk {
                if let Ok(text) = String::from_utf8(bytes.to_vec()) {
                    for line in text.lines() {
                        if let Some(data) = line.strip_prefix("data: ") {
                            if let Ok(tgi_response) = serde_json::from_str::<TGIStreamResponse>(data) {
                                let openai_chunk = serde_json::json!({
                                    "id": request_id,
                                    "object": "text_completion",
                                    "created": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "text": tgi_response.token.text
                                    }]
                                });
                                yield Ok(axum::response::sse::Event::default().data(openai_chunk.to_string()));
                            }
                        }
                    }
                }
            }
        }
        yield Ok(axum::response::sse::Event::default().data("[DONE]"));
    };
    
    Ok(sse_stream)
}

fn format_messages_as_prompt(messages: &[OpenAIMessage]) -> String {
    messages
        .iter()
        .map(|msg| format!("{}: {}", msg.role, msg.content))
        .collect::<Vec<_>>()
        .join("\n")
}

async fn classify(
    State(state): State<AppState>,
    Json(request): Json<ClassifyRequest>,
) -> Result<Json<ClassifyResponse>, StatusCode> {
    let (config, tg263_data) = state.as_ref();
    let provider = request.provider.unwrap_or_else(|| config.provider.clone());
    let top_k = request.top_k.unwrap_or(config.max_labels).min(config.max_labels).max(1);

    let raw_output = generate_classification(&provider, config, &request.input).await?;
    let normalized_output = normalize_text(&raw_output);

    let mut scores: Vec<(String, f64)> = tg263_data
        .labels
        .iter()
        .map(|label| {
            let score = strsim::jaro_winkler(&normalized_output, label);
            (label.clone(), score)
        })
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.truncate(top_k);

    let topk: Vec<ClassifyResult> = scores
        .iter()
        .filter_map(|(normalized_label, score)| {
            tg263_data.normalized_to_original.get(normalized_label).map(|original| {
                ClassifyResult {
                    label: original.clone(),
                    score: *score,
                }
            })
        })
        .collect();

    let (status, mapped_label) = if let Some(top_result) = topk.first() {
        if top_result.score >= config.abstain_sim_threshold {
            ("ok".to_string(), Some(top_result.label.clone()))
        } else {
            ("not_found".to_string(), None)
        }
    } else {
        ("not_found".to_string(), None)
    };

    Ok(Json(ClassifyResponse {
        status,
        mapped_label,
        topk,
        raw_output,
        provider,
    }))
}

async fn generate_classification(
    provider: &str,
    config: &AppConfig,
    input: &str,
) -> Result<String, StatusCode> {
    let client = reqwest::Client::new();

    match provider {
        "vllm" => {
            let chat_request = OpenAIChatRequest {
                model: Some(config.default_chat_model.clone()),
                messages: vec![
                    OpenAIMessage {
                        role: "system".to_string(),
                        content: "You are a medical structure classifier. Given an anatomical description, provide only the most likely TG-263 structure name.".to_string(),
                    },
                    OpenAIMessage {
                        role: "user".to_string(),
                        content: input.to_string(),
                    },
                ],
                temperature: Some(0.0),
                top_p: Some(1.0),
                max_tokens: Some(8),
                stream: Some(false),
                provider: None,
            };

            let url = format!("{}/v1/chat/completions", config.vllm_url);
            let response = client
                .post(&url)
                .json(&chat_request)
                .send()
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            if !response.status().is_success() {
                return Err(StatusCode::BAD_GATEWAY);
            }

            let response_json: serde_json::Value = response
                .json()
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            let content = response_json["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .to_string();

            Ok(content)
        }
        "tgi" => {
            let prompt = format!(
                "system: You are a medical structure classifier. Given an anatomical description, provide only the most likely TG-263 structure name.\nuser: {}",
                input
            );

            let tgi_request = TGIGenerateRequest {
                inputs: prompt,
                parameters: TGIParameters {
                    temperature: Some(0.0),
                    top_p: Some(1.0),
                    max_new_tokens: Some(8),
                },
                stream: Some(false),
            };

            let url = format!("{}/generate", config.tgi_url);
            let response = client
                .post(&url)
                .json(&tgi_request)
                .send()
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            if !response.status().is_success() {
                return Err(StatusCode::BAD_GATEWAY);
            }

            let response_json: TGIResponse = response
                .json()
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            Ok(response_json.generated_text)
        }
        _ => Err(StatusCode::BAD_REQUEST),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let config = AppConfig {
        provider: env::var("PROVIDER").unwrap_or_else(|_| "tgi".to_string()),
        vllm_url: env::var("VLLM_URL").unwrap_or_else(|_| "http://vllm:8000".to_string()),
        tgi_url: env::var("TGI_URL").unwrap_or_else(|_| "http://tgi:80".to_string()),
        default_chat_model: env::var("DEFAULT_CHAT_MODEL")
            .unwrap_or_else(|_| "tg263-8b".to_string()),
        tg263_csv: env::var("TG263_CSV")
            .unwrap_or_else(|_| "/app/labels/TG263.csv".to_string()),
        abstain_sim_threshold: env::var("ABSTAIN_SIM_THRESHOLD")
            .unwrap_or_else(|_| "0.85".to_string())
            .parse()
            .unwrap_or(0.85),
        max_labels: env::var("MAX_LABELS")
            .unwrap_or_else(|_| "5".to_string())
            .parse()
            .unwrap_or(5),
    };

    let tg263_data = load_tg263_data(&config.tg263_csv)?;
    let state = Arc::new((config, tg263_data));

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/classify", post(classify))
        .with_state(state);

    let listener = TcpListener::bind("0.0.0.0:8080").await?;
    info!("AnatomInd API server starting on 0.0.0.0:8080");

    axum::serve(listener, app).await?;

    Ok(())
}
