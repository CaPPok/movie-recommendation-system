# Hiện trạng triển khai — cập nhật 2026-07-29

Tài liệu này ghi lại những gì đã thực sự chạy trên AWS, khác gì so với kế hoạch
ban đầu, và ai còn phải làm gì. Mọi con số ở đây lấy từ lần kiểm tra thật trên
tài khoản, không phải ước lượng.

Đọc `docs/aws_deployment.md` để hiểu thiết kế tổng thể. **Lưu ý: tài liệu đó mô
tả kiến trúc batch-first không có endpoint, và đã lệch so với hệ thống hiện tại.**
Mục 2 dưới đây giải thích vì sao.

---

## 1. Tóm tắt

Hệ thống gợi ý đã chạy trên AWS. Backend gọi được model qua một SageMaker
real-time endpoint, và kết quả trả về khớp từng phim, từng điểm số với bản chạy
trên máy cá nhân.

| | |
|---|---|
| Endpoint | `movie-rec-endpoint` — `InService` |
| Region | `ap-southeast-1` |
| Model đang phục vụ | ALS `v1.0.1` |
| Độ trễ đo thật | 129–161 ms mỗi request |
| Bucket | `movie-recommendation-fcaj` |
| Bảng DynamoDB | 5 bảng `movie-rec-dev-*`, đã có dữ liệu |

---

## 2. Thay đổi kiến trúc: có endpoint

`docs/aws_deployment.md` viết **không dùng real-time endpoint**, vì nó tính tiền
24/7 và là khoản duy nhất có thể vỡ ngân sách. Thiết kế cũ để engine chạy
in-process trong backend.

**Quyết định ngày 2026-07-29: dùng endpoint.** Lý do:

* ngân sách còn 198/200 USD và dự án chỉ chạy hơn nửa tháng, nên chi phí endpoint
  nằm trong tầm;
* tách ML serving khỏi backend là kiến trúc rõ ràng hơn để trình bày, và là điều
  phía web đề nghị.

Hệ quả: mục 5, 7.2 và 9.2 của `aws_deployment.md` không còn mô tả đúng hệ thống.
Hai tài liệu cần được hợp nhất trước khi nộp.

```
Frontend  →  Backend  →  invoke_endpoint  →  movie-rec-endpoint
                 ↓                                   ↓
             DynamoDB                       artifact nạp sẵn trong RAM
          (metadata phim)                    (ALS + content-based)
```

Backend gom context từ DynamoDB, gửi sang endpoint, nhận về danh sách
`movie_id` + `score`, rồi tự ghép metadata từ bảng `Movies`.

---

## 3. Tài nguyên trên AWS

### S3 — `movie-recommendation-fcaj`

Toàn bộ nằm dưới tiền tố `movie-recommender/dev/`:

| Prefix | Nội dung |
|---|---|
| `data/raw/` | CSV gốc Kaggle |
| `data/processed/` | bảng đã làm sạch |
| `data/features/` | feature nội dung, bảng interaction |
| `data/splits/` | train / validation / test |
| `data/serving/` | `movies_serving`, `popular_movies`, `top_rated` |
| `artifacts/` | `collaborative/v1.0.0`, `v1.0.1`, `content_based/`, `LATEST.json` |
| `models/v1.0.1/model.tar.gz` | bundle endpoint đang dùng |

Đã bật: chặn public access (cả 4 cờ), versioning, và **lifecycle rule xoá version
cũ sau 30 ngày**. Không có rule này thì mỗi lần push đẻ thêm một bản của mọi file.

### DynamoDB — 5 bảng, on-demand

| Bảng | Khoá | Số item |
|---|---|---|
| `movie-rec-dev-Movies` | `movie_id` | 45.430 |
| `movie-rec-dev-Users` | `user_id` | 270.883 |
| `movie-rec-dev-UserInteractions` | `user_id` + `interaction_key` | 9.217 |
| `movie-rec-dev-PopularMovies` | `list_id` | 22 |
| `movie-rec-dev-RecommendationCache` | `user_id` + `scenario` | — |

9.217 interaction hiện có là rating lịch sử từ dataset (timestamp 1997), không
phải event người dùng thật.

### SageMaker

| | |
|---|---|
| Endpoint | `movie-rec-endpoint` |
| Instance | `ml.m5.large`, 1 máy |
| Image | `pytorch-inference:2.5.1-cpu-py311` |
| Execution role | `AmazonSageMaker-ExecutionRole-20260727T132467` |
| Quota `ml.m5.large` | 4 |

Vì sao image PyTorch cho một model không có PyTorch: xem mục 6.

---

## 4. Cách backend gọi endpoint

Request **phải gửi đủ context**, không chỉ `user_id` — engine có ba kịch bản và
tự chọn kịch bản dựa trên các trường này:

```json
{
  "user_id": 12345,
  "scenario_hint": "returning_user",
  "onboarding_completed": true,
  "valid_interaction_count_90d": 25,
  "selected_movie_ids": [],
  "selected_genres": ["Drama"],
  "recent_interactions": [
    {"movie_id": 862, "event_type": "rating", "value": 4.5}
  ],
  "exclude_movie_ids": [],
  "limit": 10
}
```

Response:

```json
{
  "model_name": "hybrid_recommender",
  "scenario_applied": "returning_user",
  "fallback_used": false,
  "fallback_level": "none",
  "artifact_versions": {"collaborative": "v1.0.1", "content_based": "local"},
  "recommendations": [
    {"movie_id": 240, "score": 1.0, "reason_code": "similar_users",
     "reason_context": {}}
  ]
}
```

Ba điều dễ sai:

* **Guest gửi `"user_id": null`**, không phải chuỗi `"guest"`. Engine ép kiểu
  integer và trả lỗi nếu nhận chuỗi.
* **Response không có `title`, `overview`, `poster_path`.** Backend tự
  `BatchGetItem` bảng `Movies` để ghép metadata. Đây là cố ý: model không giữ bản
  sao thứ hai của catalogue.
* **Request hỏng trả HTTP 200 kèm trường `error`**, không phải 5xx. Lỗi 5xx nghĩa
  là endpoint thật sự có vấn đề; hai trường hợp đó phải phân biệt được.

Kết quả gọi thử ngày 2026-07-29:

| Scenario | `fallback_level` | Thời gian |
|---|---|---|
| `guest` | `none` | 643 ms (lần đầu) |
| `onboarding_user` | `none` | 129 ms |
| `returning_user` | `none` | 161 ms |

`fallback_level: none` nghĩa là mỗi kịch bản được trả lời bằng đúng model dành
cho nó, không phải nguồn dự phòng.

---

## 5. Cấu hình cho backend

```env
AWS_REGION=ap-southeast-1
AWS_DEFAULT_REGION=ap-southeast-1

AWS_SAGEMAKER_ENDPOINT_NAME=movie-rec-endpoint
AWS_SAGEMAKER_INSTANCE_TYPE=ml.m5.large

AWS_DYNAMODB_MOVIES_TABLE=movie-rec-dev-Movies
AWS_DYNAMODB_POPULAR_TABLE=movie-rec-dev-PopularMovies
AWS_DYNAMODB_USERS_TABLE=movie-rec-dev-Users
AWS_DYNAMODB_INTERACTIONS_TABLE=movie-rec-dev-UserInteractions
AWS_DYNAMODB_RECOMMENDATION_CACHE_TABLE=movie-rec-dev-RecommendationCache

AWS_S3_BUCKET=movie-recommendation-fcaj
AWS_S3_MODEL_PREFIX=movie-recommender/dev/models/

JWT_SECRET_KEY=          # tự sinh, tối thiểu 32 byte
```

Role hoặc user chạy backend cần thêm quyền `sagemaker:InvokeEndpoint`.

---

## 6. Thay đổi trong repo ML

Nhánh `feat/sagemaker-endpoint`, commit `e1036fc`.

### File mới

| File | Việc |
|---|---|
| `src/features/text_vectorizer.py` | lưu/nạp vectorizer ở định dạng không phụ thuộc phiên bản |
| `deploy/recommendation_handler.py` | handler 4 hàm cho endpoint |
| `scripts/build_model_bundle.py` | đóng gói `model.tar.gz` |
| `scripts/deploy_endpoint.py` | tạo / xem trạng thái / xoá endpoint |
| `scripts/invoke_endpoint.py` | gọi thử endpoint |
| `scripts/convert_vectorizer.py` | chuyển artifact cũ sang định dạng mới |
| `requirements-endpoint.txt` | thư viện cài trong container serving |
| `deploy/sagemaker_endpoint_s3_policy.example.json` | mẫu IAM policy |

### File sửa

* `configs/aws.yaml` — prefix S3 khớp cây thật trên bucket, tên bảng
  `movie-rec-dev-UserInteractions`, sort key `interaction_key`, thêm khối
  `sagemaker.endpoint`
* `src/features/content.py` — ghi thêm vectorizer dạng phổ thông
* `src/recommenders/content_based.py` — đọc dạng phổ thông thay vì pickle

### Vì sao bỏ pickle của vectorizer

`joblib.dump(TfidfVectorizer)` sinh ra pickle của scikit-learn, mà dự án này chạy
trên **ba phiên bản khác nhau**: 1.8 ở máy cá nhân, 1.4 trong job retrain, 1.2
trong image inference mới nhất AWS có.

Đo thật: artifact tạo bởi 1.8 nạp được dưới 1.2.2 **không báo lỗi**, rồi hỏng ở
lần `transform` đầu tiên với `NotFittedError: idf vector is not fitted` — và chỉ
trên đường onboarding. Nghĩa là endpoint sẽ báo khoẻ mạnh và hỏng trên một phần
lưu lượng.

Cách sửa: lưu từ vựng, trọng số idf và tham số analyzer thành JSON + `.npy`. Dựng
lại `transform` từ ba thứ đó cho kết quả **giống hệt từng bit** — đã kiểm chứng
bằng cách so bản chạy trên 1.2.2 với bản chạy trên 1.8.0, sai lệch `0.000e+00`.

Sửa này cũng vá luôn một lỗi tương lai: job retrain hàng tuần chạy trên
scikit-learn 1.4 và trước đây sẽ sinh ra pickle mà endpoint không đọc được, tức
là **vòng lặp phản hồi sẽ tự phá endpoint mỗi tuần**.

---

## 7. Bốn lỗi đã gặp khi deploy

Ghi lại vì mỗi lỗi che lỗi sau, và ba lỗi đầu đều biểu hiện giống hệt nhau.

### Lỗi 1 — thiếu `kms:Decrypt`

Bucket mã hoá bằng khoá `aws/s3`. Role có `s3:GetObject` nên lấy được file, nhưng
không giải mã được nội dung. SageMaker không mở được bundle, container không khởi
động, **CloudWatch trống trơn**.

*Dấu hiệu nhận biết: không có log nào cả nghĩa là chết trước khi chạy, không phải
lỗi trong code.*

### Lỗi 2 — permissions boundary

`SageMakerExecutionRole` nhìn thì đủ quyền (`AmazonSageMakerFullAccess`, trust
policy đúng), nhưng có **permissions boundary** `AmazonS3FullAccess`. Boundary là
trần quyền: quyền thực tế = policy ∩ boundary. Trần đó chỉ cho S3, nên `ecr:*`,
`logs:*`, `kms:*` đều bị chặn — role **không kéo được container image**.

Đã chuyển sang `AmazonSageMaker-ExecutionRole-20260727T132467`, không có boundary.

> Luôn kiểm tra `aws iam get-role` xem `PermissionsBoundary`, chứ không chỉ
> `list-attached-role-policies`.

### Lỗi 3 — container Python quá cũ

Mọi image scikit-learn có bản inference đều dừng ở `1.2-1`, chạy **Python 3.9**.
Đường serving dùng `zip(..., strict=True)` — cú pháp chỉ có từ **Python 3.10**.

Kèm theo: `requirements.txt` khai `numpy>=1.24`, pip nâng lên numpy 2 đè lên scipy
biên dịch cho numpy 1, container chết với
`ImportError: numpy._core.multiarray failed to import`.

Đã chuyển sang image `pytorch-inference:2.5.1-cpu-py311`. Không có dòng nào import
torch; dùng image đó vì **Python 3.11**, đúng bản mọi test local đã chạy. Cả hai
họ framework dùng chung contract `model_fn / input_fn / predict_fn / output_fn`
nên handler không phải sửa.

`requirements-endpoint.txt` giờ khai numpy cùng lượt với scipy và scikit-learn để
pip chọn một bộ khớp nhau, thay vì nâng lẻ một cái.

### Lỗi 4 — tên file trùng package hệ thống

Handler đặt tên `sagemaker_inference.py`, đúng tên package mà TorchServe cần
import. `/opt/ml/code` đứng trước `site-packages` trong `sys.path`, nên Python vớ
phải file của mình:

```
ModuleNotFoundError: No module named 'sagemaker_inference.default_handler_service';
'sagemaker_inference' is not a package
```

Đã đổi thành `recommendation_handler.py`.

**Chi tiết đáng nhớ nhất:** worker chết nhưng `/ping` vẫn trả 200 đủ lâu để AWS
tuyên bố `InService`. Endpoint "sống" trên giấy tờ, và chỉ lộ ra khi có request
thật — lúc đó client nhận `ReadTimeoutError` chứ không phải lỗi tử tế.

> Đừng tin trạng thái `InService`. Luôn gọi thử một request thật rồi mới kết luận.

---

## 8. Việc còn lại

### Phần model (Hiệp)

* **Sửa `scripts/export_interactions.py`** để đọc đúng schema backend đang ghi.
  Backend ghi bộ ba `(interaction_type, interaction_action, interaction_value)`;
  model cần `event_type` + `value`. Quy đổi được 7/8 loại event:

  | Backend | → model |
  |---|---|
  | `click`, `record`, `1` | `click` |
  | `watch`, `record`, `0.0–1.0` | `watch` (hoặc `complete` nếu ≥ 0.95) |
  | `rating`, `set`, `0.5–5.0` | `rating` |
  | `reaction`, `set`, `1` | `like` |
  | `reaction`, `set`, `-1` | `dislike` |
  | `share`, `record`, `1` | `share` |
  | `*`, `clear`, `0` | bỏ |

  Cũng phải ép `user_id` và `movie_id` từ chuỗi sang số.

* **Hợp nhất `aws_deployment.md` với tài liệu này** — hiện hai bên mô tả hai kiến
  trúc khác nhau.
* Retrain bằng SageMaker Processing Job: `--dry-run`, smoke test, rồi chạy thật.
* IAM role cho Processing Job (role endpoint chỉ có quyền đọc).

### Phần web (Ái)

* Đổi endpoint backend sang `POST /model/recommend` — `recent_interactions` không
  truyền qua query string được.
* Bỏ `title` khỏi response của model, tự ghép metadata từ bảng `Movies`.
* Validate `event_type` theo danh sách hợp lệ; guest gửi `user_id: null`.
* Frontend bắn event khi: mở trang chi tiết, xem qua 50%, xem hết, thích/không
  thích, chấm sao, chia sẻ.
* **`comment`** là loại event duy nhất model hỗ trợ mà backend chưa có. Cần ô bình
  luận cộng phân tích sentiment. Model vẫn chạy tốt khi thiếu nó.

---

## 9. Chi phí và dọn dẹp

Endpoint **tính tiền theo giờ kể từ lúc tạo, kể cả khi không ai gọi**. Đây là
khoản duy nhất trong hệ thống chạy 24/7; mọi thứ còn lại (S3 ~2 GB, DynamoDB
on-demand) ở mức vài xu tới vài USD mỗi tháng.

Budget alarm `My-200$-budget` đã bật ở mức 200 USD.

Xem trạng thái:

```bash
python scripts/deploy_endpoint.py --status
```

Xoá khi kết thúc dự án — một lệnh xoá cả endpoint, endpoint-config và model. Xoá
thiếu một trong ba thì lần deploy sau sẽ trùng tên và thất bại:

```bash
python scripts/deploy_endpoint.py --delete
```

**Đặt ngày xoá vào lịch nhóm ngay từ bây giờ.** Kiểm tra hằng tuần:

```bash
aws sagemaker list-endpoints --region ap-southeast-1
```

---

## 10. Chạy lại từ đầu

Khi cần dựng lại endpoint, ví dụ sau một lần retrain:

```bash
python scripts/build_model_bundle.py --upload
```

```bash
export MOVIE_REC_SAGEMAKER_ROLE=<arn role không có permissions boundary>
python scripts/deploy_endpoint.py --dry-run
```

```bash
python scripts/deploy_endpoint.py
```

```bash
python scripts/invoke_endpoint.py --demo
```

Bước cuối là bắt buộc, không phải tuỳ chọn — xem lỗi 4 ở mục 7.
