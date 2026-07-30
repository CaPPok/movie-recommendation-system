# Hiện trạng triển khai — cập nhật 2026-07-30

Tài liệu này ghi lại những gì đã thực sự chạy trên AWS và ai còn phải làm gì.
Mọi con số ở đây lấy từ lần kiểm tra thật trên tài khoản, không phải ước lượng.

---

## 1. Tóm tắt

Hệ thống gợi ý đã chạy được trên AWS: ngày 2026-07-29 endpoint đã lên
`InService` và trả về kết quả khớp từng phim, từng điểm số với bản chạy trên máy
cá nhân.

| | |
|---|---|
| Endpoint | `movie-rec-endpoint` — **dựng theo yêu cầu**, hiện đang tắt |
| Region | `ap-southeast-1` |
| Model đang phục vụ | ALS `v0.0.0-202607301039` — do job retrain trên SageMaker thăng cấp ngày 30/07 |
| Độ trễ đo thật | 129–161 ms mỗi request |
| Bucket | `movie-recommendation-fcaj` — 87 object, 2,3 GiB |
| Bảng DynamoDB | 5 bảng `movie-rec-dev-*`, đã có dữ liệu |
| Vòng lặp retrain | **đã chạy trọn vẹn trên AWS** — xem mục 12 |

> [!IMPORTANT]
> **Endpoint không để chạy thường trực.** Nó tính tiền theo giờ kể cả khi không
> ai gọi, và ở quy mô đồ án đó là khoản duy nhất đáng kể — mọi thứ còn lại cộng
> lại chưa tới một đô mỗi tháng. Bundle, artifact và cấu hình đều nằm sẵn trên
> S3, nên dựng lại chỉ mất một lệnh và khoảng 10 phút:
>
> ```bash
> python scripts/deploy_endpoint.py          # dựng
> python scripts/invoke_endpoint.py --demo   # kiểm tra thật, đừng tin InService
> python scripts/deploy_endpoint.py --delete # tắt khi xong
> ```
>
> Bật trước buổi demo hoặc trước khi Ái cần test, tắt ngay sau đó.

---

## 2. Kiến trúc

Theo đúng thiết kế đã thống nhất: model được train ngoài luồng, artifact nằm trên
S3, và backend chỉ gọi endpoint chứ không train và không tự chạy model.

```
Frontend  →  Backend  →  invoke_endpoint  →  movie-rec-endpoint
                 ↓                                   ↑
             DynamoDB                          model.tar.gz
          (metadata phim)                    (S3, nạp lúc khởi động)
```

Luồng một request:

1. Backend gom context của user từ DynamoDB — đã onboarding chưa, bao nhiêu tương
   tác gần đây, thể loại đã chọn.
2. Gửi sang endpoint bằng `invoke_endpoint`.
3. Endpoint trả về danh sách `movie_id` kèm `score` và `reason_code`.
4. Backend `BatchGetItem` bảng `Movies` để ghép `title`, `poster_path`, rồi trả
   cho frontend.

Endpoint nạp artifact vào RAM một lần lúc khởi động và trả lời từ bộ nhớ, không
đọc S3 lại mỗi request. Đo thật: 129–161 ms một lượt.

Model bên trong là **hybrid** — ALS cho user có lịch sử, content-based TF-IDF cho
user mới, và bảng xếp hạng phổ biến cho khách chưa đăng nhập. Endpoint tự chọn
nhánh nào dựa trên context backend gửi lên, và báo lại đã chọn nhánh nào qua
trường `scenario_applied`.

---

## 3. Tài nguyên trên AWS

### S3 — `movie-recommendation-fcaj`

Cây tổ chức theo **giai đoạn vòng đời** ở gốc bucket. Bố cục chi tiết từng file
kèm công dụng nằm ở [`S3_STRUCTURE.txt`](../S3_STRUCTURE.txt).

| Prefix | Nội dung | Số file |
|---|---|---:|
| `datasets/raw/` | CSV gốc Kaggle | 5 |
| `datasets/processed/` | bảng đã làm sạch + feature | 12 |
| `datasets/serving/` | export DynamoDB của phía web, model không đọc | 5 |
| `datasets/exports/` | event xuất từ DynamoDB để retrain | 1 |
| `training/` | train / validation / test | 3 |
| `models/` | `collaborative/v1.0.0`, `v1.0.1`, `content_based/`, `eval/`, `LATEST.json` | 22 |
| `models/bundles/<version>/` | `model.tar.gz` cho endpoint | 2 |
| `inference/` | `movies_serving`, `popular_movies`, `top_rated_*` | 5 |
| `evaluation/` | báo cáo JSON | 15 |
| `logs/` | log chạy | 2 |

> [!WARNING]
> Trước 30/07 file này ghi mọi prefix nằm dưới `movie-recommender/dev/`.
> **Tiền tố đó chưa bao giờ tồn tại trên bucket.** Hệ quả: `push` sẽ đẻ ra một cây
> song song thứ hai, còn `pull` không tìm thấy gì và job retrain train trên dữ
> liệu rỗng. Đã sửa `configs/aws.yaml` trỏ về cây thật.

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
| Endpoint | `movie-rec-endpoint` — dựng theo yêu cầu |
| Instance | `ml.m5.large`, 1 máy |
| Image | `pytorch-inference:2.5.1-cpu-py311` |
| Execution role | `AmazonSageMaker-ExecutionRole-20260727T132467` |
| Quota endpoint `ml.m5.large` | 4 |
| Quota processing / training | **0** — đã xin tăng, xem mục 9 |

Vì sao image PyTorch cho một model không có PyTorch: xem mục 6.

Hai policy đã gắn vào execution role, cả hai chỉ giới hạn trong bucket của dự án:

| Policy | Quyền | Dùng cho |
|---|---|---|
| `MovieRecBucketRead` | `s3:GetObject`, `s3:ListBucket`, `kms:Decrypt` | endpoint tải `model.tar.gz` |
| `MovieRecBucketWrite` | `s3:PutObject`, `s3:DeleteObject`, `kms:GenerateDataKey` | job retrain đẩy artifact mới lên |

`kms:*` là bắt buộc chứ không thừa: bucket mã hoá bằng khoá `aws/s3`, và thiếu nó
thì đọc được vỏ file nhưng không mở được nội dung — xem lỗi 1 ở mục 7.

### Không thuộc phần model

Có một EC2 `t3.micro` đang chạy trong region này. Nó **không phải** của phần
model — nhiều khả năng là máy chủ backend. Đừng tắt nó khi dọn dẹp chi phí phía
ML; hỏi phía web trước.

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
AWS_S3_DATASET_PREFIX=datasets/raw/
AWS_S3_PROCESSED_PREFIX=datasets/processed/
AWS_S3_SERVING_PREFIX=inference/
AWS_S3_TRAINING_PREFIX=training/
AWS_S3_MODEL_PREFIX=models/bundles/
AWS_S3_OUTPUT_PREFIX=evaluation/

JWT_SECRET_KEY=          # tự sinh, tối thiểu 32 byte
```

> [!WARNING]
> Sáu giá trị prefix trên **đã đổi ngày 30/07**. Bản cũ ghi
> `movie-recommender/dev/...`, một tiền tố chưa bao giờ tồn tại trên bucket. Nếu
> backend đã copy bản cũ thì phải cập nhật lại, xem mục 12.2.

Chỉ `AWS_S3_BUCKET` là bắt buộc — backend hiện không gọi S3, nó chỉ validate tên
bucket lúc khởi động. Sáu prefix còn lại điền sẵn để cấu hình không phải đoán khi
có thành phần đọc S3 thật.

Lưu ý `AWS_S3_SERVING_PREFIX` trỏ `inference/` chứ không phải `datasets/serving/`.
Hai chỗ đó khác nhau: `datasets/serving/` là export DynamoDB do phía web tạo, còn
`inference/` là bảng tra cứu do pipeline ML sinh ra (`movies_serving`,
`top_rated_*`).

`models/` (artifact thô) không có biến tương ứng vì backend không đọc tới —
endpoint đã mang sẵn bản sao bên trong `model.tar.gz`. `AWS_S3_MODEL_PREFIX` trỏ
`models/bundles/` là nơi chứa các `model.tar.gz` theo version.

Role hoặc user chạy backend cần thêm quyền `sagemaker:InvokeEndpoint`.

---

## 6. Thay đổi trong repo ML

Nhánh `feat/sagemaker-endpoint`.

### File mới

| File | Việc |
|---|---|
| `src/features/text_vectorizer.py` | lưu/nạp vectorizer ở định dạng không phụ thuộc phiên bản |
| `deploy/recommendation_handler.py` | handler 4 hàm cho endpoint |
| `scripts/build_model_bundle.py` | đóng gói `model.tar.gz` |
| `scripts/deploy_endpoint.py` | tạo / xem trạng thái / xoá / cập nhật endpoint |
| `scripts/invoke_endpoint.py` | gọi thử endpoint |
| `scripts/convert_vectorizer.py` | chuyển artifact cũ sang định dạng mới |
| `requirements-endpoint.txt` | thư viện cài trong container serving |
| `tests/test_export_translation.py` | khoá bảng quy đổi event backend → model |
| `deploy/*_policy.example.json` | mẫu IAM policy cho endpoint và cho job retrain |

### File sửa

* `configs/aws.yaml` — prefix S3 khớp cây thật trên bucket, tên bảng
  `movie-rec-dev-UserInteractions`, sort key `interaction_key`, ngưỡng
  `watch_complete_threshold`, thêm khối `sagemaker.endpoint`
* `src/features/content.py` — ghi thêm vectorizer dạng phổ thông
* `src/recommenders/content_based.py` — đọc dạng phổ thông thay vì pickle
* `scripts/export_interactions.py` — quy đổi schema backend sang event của model
* `deploy/sagemaker_retrain.py` — đóng gói bundle sau khi thăng cấp
* `scripts/sagemaker_retrain_job.py` — mặc định bật `--build-bundle`

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

* ~~**Sửa `scripts/export_interactions.py`**~~ — **xong 30/07, đã chạy thật.**
  Quét 9.217 item, xuất 9.160 event, loại đúng 57 thao tác gỡ rating/reaction.
  Bảng quy đổi bên dưới đã hoạt động trên dữ liệu thật:

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

* ~~Chờ AWS duyệt quota~~ — **đã có quota, job retrain đã chạy xong 30/07.**
* Đặt lịch chạy tự động — vẫn chưa làm. Chỉ nên làm sau khi có đủ event thật;
  hiện 9.057/9.160 event là rating lịch sử seed lại, xem mục 12.

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

## 9. Vòng lặp retrain

Train lần đầu chạy ở máy cá nhân — ALS mất 31 giây, không có lý do trả tiền cho
việc đó. **Train lại định kỳ thì chạy trên AWS**, vì nó phải chạy được khi không
ai ngồi ở máy.

```
DynamoDB UserInteractions
      ↓  scripts/export_interactions.py --upload
S3 events/
      ↓  scripts/sagemaker_retrain_job.py --version vX.Y.Z --events s3://.../events/
SageMaker Processing Job
      ├─ nạp event, dựng lại split, train ALS
      ├─ đánh giá và chạy cổng kiểm duyệt
      ├─ đạt  → đẩy artifact + LATEST.json lên S3, rồi đóng gói model.tar.gz
      └─ không đạt → giữ nguyên LATEST.json, job vẫn kết thúc thành công
      ↓
python scripts/deploy_endpoint.py     ← cập nhật endpoint tại chỗ
```

Bước cuối là thủ công có chủ đích: model mới thay model đang phục vụ người dùng
thật, nên có một người bấm nút.

### Cổng kiểm duyệt

Retrain chạy theo lịch nghĩa là không ai xem lại từng lần chạy. Ba điều kiện
trong `configs/aws.yaml` khối `retraining.promotion`, mỗi điều kiện chỉ có quyền
**chặn**:

| Điều kiện | Ý nghĩa |
|---|---|
| tối thiểu 1.000 user được chấm | dưới mức đó chỉ số là nhiễu, không phải phép đo |
| thắng baseline phổ biến | model cá nhân hoá không thắng nổi "ai cũng xem phim hot" thì không đáng phục vụ |
| không tụt quá 5% | dung sai, không phải yêu cầu lần nào cũng phải tốt hơn |

Không đạt thì `LATEST.json` giữ nguyên, artifact mới vẫn lưu để xem xét, và job
vẫn kết thúc **thành công**. Cổng chặn là cổng làm đúng việc, không phải sự cố.
Job cũng bỏ qua luôn bước đóng gói, vì endpoint không có gì mới để phục vụ.

### Quy đổi event

Backend ghi bộ ba `(interaction_type, interaction_action, interaction_value)`;
model chấm điểm theo tám loại event phẳng. `scripts/export_interactions.py` quy
đổi, và `tests/test_export_translation.py` khoá bảng ánh xạ lại:

| Backend | → model |
|---|---|
| `click` / `share`, `record`, `1` | `click` / `share` |
| `watch`, `record`, `< 0.95` | `watch`, kèm tỉ lệ đã xem |
| `watch`, `record`, `≥ 0.95` | `complete` |
| `rating`, `set`, `0.5–5.0` | `rating`, kèm số sao |
| `reaction`, `set`, `1` / `-1` | `like` / `dislike` |
| bất kỳ, `clear`, `0` | bỏ — người dùng gỡ đánh giá, không phải tín hiệu âm |

`comment` là loại duy nhất model hỗ trợ mà backend chưa sinh ra.

Có một lỗi im lặng ở đây đáng biết: đọc sai tên trường thì mọi dòng xuất ra có
`event_type: null`, file vẫn được ghi, và retrain **chạy trên dữ liệu rỗng mà
không báo lỗi**. Đó là lý do bảng ánh xạ có test riêng.

### Quota — đã xong

Yêu cầu tăng quota gửi ngày 2026-07-29 **đã được duyệt**. Job retrain đầu tiên
chạy thành công ngày 30/07 trên `ml.m5.xlarge`, xem mục 12.

Lệnh tra lại lịch sử yêu cầu nếu cần:

```bash
aws service-quotas list-requested-service-quota-change-history \
  --service-code sagemaker --region ap-southeast-1
```

Khi quota về, chạy thử bằng máy nhỏ trước:

```bash
python scripts/sagemaker_retrain_job.py --version v1.1.0-smoke --instance-type ml.m5.large --wait
```

Dòng log đầu tiên in ra phiên bản Python của container. Đây là thứ duy nhất
không xác minh được từ máy local, và cũng chính là thứ đã làm endpoint chết ở lần
deploy thứ ba — xem lỗi 3 ở mục 7. Nếu nó dưới 3.10 thì job sẽ hỏng ở
`zip(..., strict=True)`, và phải đổi image giống như đã làm với endpoint.

---

## 10. Chi phí và dọn dẹp

Giá lấy từ AWS Pricing API ngày 2026-07-29 cho `ap-southeast-1`. Giá thay đổi
theo thời điểm và region; kiểm tra lại trước khi cam kết ngân sách.

### Endpoint — khoản duy nhất đáng kể

`ml.m5.large` hosting: **0,144 USD/giờ**, tính từ lúc tạo tới lúc xoá, không phụ
thuộc có ai gọi hay không.

| Để chạy | Chi phí |
|---|---|
| 1 giờ | 0,14 USD |
| 1 ngày | 3,46 USD |
| 1 tuần | 24,19 USD |
| 30 ngày | 103,68 USD |

Cách dùng quyết định con số cuối cùng nhiều hơn là bản thân giá:

| Cách dùng trong nửa tháng | Chi phí |
|---|---|
| Bật liên tục | ~52 USD |
| Bật 8 giờ mỗi ngày | ~17 USD |
| Chỉ bật lúc demo, ~2 giờ/ngày | ~4 USD |

### Mọi thứ còn lại

S3 khoảng 2 GB, DynamoDB on-demand, CloudWatch Logs — cộng lại **dưới 1 USD mỗi
tháng** ở quy mô này. Không đáng để dọn, và xoá thì mất dữ liệu phải nạp lại.

Job retrain tính tiền theo giây và chỉ khi chạy: một lần khoảng 10 phút trên
`ml.m5.xlarge` rơi vào cỡ vài cent.

### Cảnh báo đã bật

Budget `My-200$-budget`, giới hạn 200 USD/tháng, báo về email của cả nhóm ở các
mốc thực chi 25, 50, **75**, 100, 150 USD và mốc dự báo 200 USD.

### Lệnh cần nhớ

```bash
python scripts/deploy_endpoint.py --status
```

```bash
python scripts/deploy_endpoint.py --delete
```

`--delete` xoá cả ba: endpoint, endpoint-config và model. Xoá thiếu một trong ba
thì lần deploy sau trùng tên và thất bại.

Kiểm tra hằng tuần — kết quả nên rỗng trừ lúc đang demo:

```bash
aws sagemaker list-endpoints --region ap-southeast-1
```

---

## 11. Chạy lại từ đầu

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

---

## 12. Nhật ký 2026-07-30

Ghi lại theo thứ tự phát hiện, để lần sau nhìn vào biết chuyện gì đã xảy ra.

### 12.1. Job retrain đầu tiên chạy trọn vẹn trên AWS

```
movie-rec-retrain-20260730-103746
Status    : Completed
Chạy      : 17:38:26 → 17:45:21  (6 phút 55 giây)
Instance  : ml.m5.xlarge
```

Cổng kiểm duyệt **cho qua** và `LATEST.json` trên S3 chuyển từ `v1.0.1` sang
`v0.0.0-202607301039`. Đây là lần đầu vòng lặp retrain khép kín trên cloud.

Bundle tương ứng đã có sẵn: `models/bundles/v0.0.0-202607301039/model.tar.gz`.

### 12.2. Prefix S3 trỏ vào chỗ không tồn tại

`configs/aws.yaml` khai mọi prefix dưới `movie-recommender/dev/`, kèm ghi chú
"RESOLVED 2026-07-29: tree đó đã có sẵn splits và content-based artifacts". Liệt
kê bucket cho thấy **tiền tố đó không tồn tại**.

Hậu quả nếu không phát hiện: `push` đẻ cây song song thứ hai, `pull` không thấy
gì và job retrain train trên dữ liệu rỗng — mà **không báo lỗi**.

Đã sửa: prefix trỏ về cây thật, và lấp các thư mục còn rỗng. Trước khi sửa, ba
thứ sau **hoàn toàn chưa có trên S3**:

* `training/` — không có split thì không train được
* `models/` — không có artifact thì backend không serve được, và cổng kiểm duyệt
  mất tác dụng vì không có model cũ để so
* `top_rated_all.parquet`, `top_rated_by_genre.parquet` — thiếu thì kịch bản
  khách vãng lai lỗi

### 12.3. Luồng DynamoDB → S3 đã chạy thật

```bash
python scripts/export_interactions.py --upload
```

```
Scan movie-rec-dev-UserInteractions  →  9.160 event
                                        57 thao tác gỡ, đã loại
→ s3://movie-recommendation-fcaj/datasets/exports/2026-07-30.jsonl  (1.009,3 KiB)
```

Đọc ngược từ S3 bằng chính hàm `retrain.py` dùng: ra lại đủ 9.160 event, đủ 5
trường. Vòng ghi/đọc khép kín.

### 12.4. Backend đang ghi thiếu `value` — 26 event bị vứt

Chạy 103 event thật (2026) qua đúng bộ chấm điểm của model:

```
events_received : 103
events_counted  :  71
events_ignored  :  26   →  {'malformed_value': 26}
```

Chi tiết theo loại, nhóm event 2026:

| Event | Tổng | Thiếu `value` |
|---|---:|---:|
| `watch` | 13 | **13 (100%)** |
| `rating` | 41 | **13** |
| `click` | 28 | 5 |
| `like` | 7 | 0 |
| `share` | 14 | 0 |

Đối chiếu: **9.057 event lịch sử có `value` đủ 100%.** Nên đây không phải lỗi
export mà là backend ghi thiếu. `watch` không có `value` thì không biết xem bao
nhiêu phần trăm, mà ngưỡng là 50% → bị vứt toàn bộ.

**Việc cho phía web:** khi ghi vào `movie-rec-dev-UserInteractions`, bắt buộc kèm
`value` — `watch` là tỉ lệ đã xem (0–1), `rating` là số sao. `like`/`share`
không cần và hiện đang làm đúng.

### 12.5. Dữ liệu trong bảng gần như toàn là lịch sử seed lại

| Nguồn | Số event |
|---|---:|
| Lịch sử 1996–2017 (MovieLens seed vào Dynamo) | 9.057 |
| **Tương tác thật 2026** | **103** |

103 event đó đến từ **đúng 1 user**. Nạp 9.057 event lịch sử vào `retrain.py` là
nạp lại chính dữ liệu đã có trong tập train; chính sách `prefer_rating` khử trùng
nên không hỏng, nhưng cũng không thêm gì. Đây là lý do chưa nên đặt lịch retrain
tự động.

### 12.6. SDK đổ file vào gốc bucket

SageMaker SDK tự chọn chỗ upload nếu không được chỉ định:

| Nguồn | Cái gì | Đi đâu |
|---|---|---|
| `PyTorchModel(...)` thiếu `code_location` | `sourcedir.tar.gz` | `s3://bucket/<model_name>/` |
| `Session(...)` thiếu `default_bucket_prefix` | `source/sourcedir.tar.gz`, `runproc.sh` | `s3://bucket/<job_name>/` |

Tám lần chạy từ 29/07 để lại tám thư mục ở gốc, chiếm **707,2 MB** dưới dạng
version cũ và delete marker — vẫn tính tiền mà không hiện khi liệt kê thường.

Đã sửa: `deploy_endpoint.py` truyền `code_location` trỏ vào
`models/bundles/<version>/`, `sagemaker_retrain_job.py` đặt
`default_bucket_prefix = "training/jobs"`. Đã xoá sạch 707,2 MB rác.

Gốc bucket giờ chỉ còn 6 thư mục đúng thiết kế.

### 12.7. Bẫy hay gặp: `load_env.ps1` chỉ có tác dụng trong một cửa sổ

`deploy_endpoint.py` báo `Chưa có SageMaker execution role` dù sáng cùng ngày job
vẫn chạy được. Nguyên nhân: `role_arn` để trống trong `configs/aws.yaml` (cố ý,
không commit ARN vào git), giá trị thật nằm trong `.env`, mà `load_env.ps1` chỉ
nạp vào **cửa sổ PowerShell đang chạy nó**.

Mở cửa sổ mới thì phải chạy lại:

```powershell
.\load_env.ps1
```

Muốn khỏi phải nhớ thì đặt biến vĩnh viễn cho user, rồi mở cửa sổ mới:

```powershell
[Environment]::SetEnvironmentVariable("MOVIE_REC_SAGEMAKER_ROLE", "<arn>", "User")
```

Đừng điền thẳng vào `configs/aws.yaml` — file đó nằm trong git, ARN kèm account
id sẽ lộ lên GitHub.

### 12.8. Còn tồn trên S3

* `models/content_based/.gitkeep` — file giữ chỗ của git bị đẩy lên nhầm
* Bốn object 0 byte tên rỗng ở `datasets/exports/`, `evaluation/`, `inference/`,
  `models/`, `training/` — marker thư mục do tạo cây thủ công để lại
* `datasets/processed/` gánh cả `processed` lẫn `features`, nên `pull` tải trùng
  242 MB bảng đặc trưng. Không hỏng, chỉ chậm.
