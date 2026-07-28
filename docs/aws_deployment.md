# Triển khai lên AWS: đưa data, train, re-train và nối vào web xem phim

Tài liệu này mô tả toàn bộ đường đi từ máy cá nhân lên AWS, và cách repo web gọi
được module gợi ý. Đọc từ trên xuống là dựng được hạ tầng từ con số không.

Kiến trúc là **batch-first**: mọi tính toán nặng chạy theo lô, kết quả ghi thành
artifact tĩnh trên S3 và bảng tra cứu trên DynamoDB. **Không có SageMaker
real-time endpoint.** Đó là khoản chi phí duy nhất có thể vượt ngân sách 100 USD
(`MODEL_DESIGN_SPEC.md` mục 2 và 4.4), và thiết kế này không dùng tới nó.

Số đo trong tài liệu lấy từ lần chạy đầy đủ ngày 2026-07-28 trên máy local, không
phải ước lượng.

## Mục lục

1. [Toàn cảnh](#1-toàn-cảnh)
2. [Chuẩn bị tài nguyên AWS](#2-chuẩn-bị-tài-nguyên-aws)
3. [Đưa data lên S3](#3-đưa-data-lên-s3)
4. [Train lần đầu](#4-train-lần-đầu)
5. [Re-train trên SageMaker](#5-re-train-trên-sagemaker-processing-job)
6. [Re-train trên EC2](#6-re-train-trên-ec2)
7. [Nối vào repo web](#7-nối-vào-repo-web)
8. [Vòng lặp phản hồi](#8-vòng-lặp-phản-hồi-đầy-đủ)
9. [Chi phí và cách kiểm soát](#9-chi-phí-và-cách-kiểm-soát)
10. [Vận hành, giám sát và quay lui](#10-vận-hành-giám-sát-và-quay-lui)
11. [Dọn dẹp khi kết thúc](#11-dọn-dẹp-khi-kết-thúc)
12. [Checklist](#12-checklist-triển-khai)

---

## 1. Toàn cảnh

```
        MÁY CÁ NHÂN                          AWS                      NGƯỜI DÙNG
   ┌──────────────────┐
   │ Kaggle dataset   │
   │        ↓         │
   │ run_data_pipeline│
   │        ↓         │
   │ data/processed   │──── aws_sync.py push ────►┌──────────────────┐
   │ data/splits      │                           │  S3   bucket     │
   │ data/serving     │                           │  processed/      │
   └──────────────────┘                           │  splits/         │
                                                  │  serving/        │
   ┌──────────────────┐                           │  artifacts/      │
   │ train.py (local) │──── aws_sync.py push ────►│    collaborative/│
   └──────────────────┘                           │    content_based/│
                                                  │    LATEST.json   │
        ĐỊNH KỲ                                   │  events/         │
   ┌──────────────────┐      pull  ▲   ▼ push     │  reports/        │
   │ SageMaker        │◄───────────┴───┴──────────┤                  │
   │ Processing Job   │                           └──────────────────┘
   │      HOẶC        │                                    │
   │ EC2 + systemd    │                                    │ tải 1 lần
   │   retrain.py     │                                    │ lúc khởi động
   └──────────────────┘                                    ▼
            ▲                                    ┌──────────────────┐
            │ events/*.jsonl                     │  Backend FastAPI │
            │                                    │  (App Runner /   │
   ┌──────────────────┐                          │   ECS / EC2)     │
   │ DynamoDB         │                          │                  │
   │  Interactions    │◄─────── ghi event ───────┤ RecommendationEngine
   │  Movies          │──────── đọc metadata ───►│                  │
   │  PopularMovies   │                          └──────────────────┘
   │  Users           │                                   ▲ │
   └──────────────────┘                                   │ ▼
                                                   ┌──────────────┐
                                                   │  Frontend    │
                                                   │  React       │
                                                   └──────────────┘
```

**Cái gì chạy ở đâu, và vì sao:**

| Việc | Chạy ở | Lý do |
|---|---|---|
| Làm sạch dữ liệu, tạo feature, chia split | Máy cá nhân | Chạy một lần, mất vài phút, không cần trả tiền cloud. |
| Train lần đầu | Máy cá nhân | ALS full dataset mất 31 giây. Không có lý do gì để trả tiền. |
| Re-train định kỳ | SageMaker Processing Job **hoặc** EC2 | Cần chạy tự động khi không ai ngồi máy. |
| Suy luận (gợi ý) | Trong tiến trình backend | Artifact nằm sẵn trong RAM; thêm một network hop cho mỗi lần tải trang là thứ kiến trúc này sinh ra để tránh. |
| Lưu event, metadata phim | DynamoDB | |

> [!IMPORTANT]
> Bước train **chạy local được hoàn toàn**. Đưa lên SageMaker là để đáp ứng yêu
> cầu dùng AWS của đề tài, không phải vì local không đủ sức
> (`MODEL_DESIGN_SPEC.md` mục 4.1).

---

## 2. Chuẩn bị tài nguyên AWS

### 2.1. Hai việc làm trước tiên

Hai việc này hoặc chờ lâu, hoặc phòng ngừa thiệt hại. Làm trước khi tạo bất cứ
tài nguyên nào.

**a) Xin quota SageMaker.** Tài khoản AWS mới thường có quota **0** cho instance
SageMaker. Phát hiện chuyện này lúc sắp chạy job là mất vài ngày chờ duyệt.

```
Service Quotas → AWS services → Amazon SageMaker
→ tìm "ml.m5.xlarge for processing job usage"
```

Bằng 0 thì bấm **Request increase**, xin 2. Xin luôn
`ml.m5.large for processing job usage` để chạy smoke test rẻ hơn.

**b) Dựng hàng rào ngân sách.** Làm trước khi tạo tài nguyên, để nếu có gì chạy
sai thì biết trong vòng một ngày chứ không phải cuối tháng.

```
Billing and Cost Management → Budgets → Create budget
→ Cost budget → Monthly → đặt ngưỡng (ví dụ 10 USD)
→ Alert khi actual > 50% và > 80%, gửi về email của bạn
```

Bật thêm Cost Explorer (`Billing → Cost Explorer → Enable`). Nó mất 24 giờ để có
dữ liệu đầu tiên nên phải bật sớm.

### 2.2. Region: `ap-southeast-1` (Singapore)

**Đã chốt ngày 2026-07-27.** Singapore là region gần Việt Nam nhất, nên cũng cho
độ trễ thấp nhất tới người dùng.

Quyết định này **thay thế** hai giá trị đang tồn tại trong dự án, và cả hai phải
được sửa lại cho khớp:

| Nơi | Đang là | Phải thành |
|---|---|---|
| `configs/aws.yaml` | — | `ap-southeast-1` (**đã sửa**) |
| `deploy/movie-rec-retrain.service` | — | `ap-southeast-1` (**đã sửa**) |
| `backend/app/services/dynamodb/user_activity_repository.py` | `us-east-1` hardcode | đọc từ biến môi trường, mặc định `ap-southeast-1` |
| `backend/app/core/config.py` | không có | thêm `AWS_REGION` |
| `PROJECT_PLAN.md` | `ap-southeast-2` | `ap-southeast-1` |

Tạo bảng ở một region rồi trỏ code sang region khác thì gọi API sẽ lỗi, hoặc tệ
hơn là tự tạo bảng thừa ở region thứ hai và bị tính tiền hai lần
(`MODEL_DESIGN_SPEC.md` mục 16.9).

Đặt biến môi trường ở mọi máy và mọi container — **không hardcode**:

```bash
export AWS_REGION=ap-southeast-1
export MOVIE_REC_BUCKET=movie-recommendation-fcaj
```

Trong PowerShell:

```powershell
$env:AWS_REGION = "ap-southeast-1"
$env:MOVIE_REC_BUCKET = "movie-recommendation-fcaj"
```

### 2.3. Credentials

```bash
aws --version
```

Chưa có thì cài AWS CLI v2. Cấu hình — **tự nhập key**, đừng dán vào file trong
repo:

```bash
aws configure
```

Region `ap-southeast-1`, output `json`.

**Kiểm tra:**

```bash
aws sts get-caller-identity
```

Ghi lại `Account` trong kết quả, các bước IAM cần account id.

### 2.4. Tạo S3 bucket

Tên bucket là duy nhất toàn cầu. Nếu `movie-recommendation-fcaj` đã bị người khác
lấy, đổi tên rồi sửa `aws.bucket` trong `configs/aws.yaml`.

```bash
aws s3 mb s3://movie-recommendation-fcaj --region $AWS_REGION
```

Chặn public access và bật versioning — versioning để lỡ đè nhầm artifact còn
khôi phục được:

```bash
aws s3api put-public-access-block --bucket movie-recommendation-fcaj \
  --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

```bash
aws s3api put-bucket-versioning --bucket movie-recommendation-fcaj \
  --versioning-configuration Status=Enabled
```

**Lifecycle rule — bắt buộc nếu đã bật versioning.** Không có nó, mỗi lần
`aws_sync.py push` ghi đè sẽ tạo một version mới của **mọi file**, và version cũ
vẫn tính tiền lưu trữ. Sau 10 lần retrain là 10 bản của gần 1 GB. Tạo
`lifecycle.json`:

```json
{
  "Rules": [
    {
      "ID": "expire-old-versions",
      "Status": "Enabled",
      "Filter": {},
      "NoncurrentVersionExpiration": { "NoncurrentDays": 30 },
      "AbortIncompleteMultipartUpload": { "DaysAfterInitiation": 7 }
    }
  ]
}
```

```bash
aws s3api put-bucket-lifecycle-configuration --bucket movie-recommendation-fcaj \
  --lifecycle-configuration file://lifecycle.json
```

Version cũ tự xoá sau 30 ngày — vẫn đủ dài để quay lui model, đủ ngắn để không
tích tiền. Rule thứ hai dọn phần upload dở dang: một lần upload gần 1 GB bị đứt
để lại mảnh vụn vẫn tính tiền mà không hiện trong danh sách file.

**Kiểm tra:**

```bash
aws s3api get-bucket-lifecycle-configuration --bucket movie-recommendation-fcaj
```

Bố cục bên trong bucket (khớp `MODEL_DESIGN_SPEC.md` mục 14.1 và
`configs/aws.yaml`):

```
s3://movie-recommendation-fcaj/
├── raw/            # CSV gốc từ Kaggle (tùy chọn, chỉ để lưu vết)
├── processed/      # data/processed  — bảng đã làm sạch
├── features/       # data/features   — feature nội dung + bảng interaction
├── splits/         # data/splits     — train / validation / test
├── serving/        # data/serving    — movies_serving, top_rated
├── artifacts/
│   ├── content_based/
│   ├── collaborative/v1.0.0/
│   └── LATEST.json
├── events/         # export interaction từ DynamoDB, mỗi lần một file JSONL
└── reports/        # báo cáo validation và retrain
```

### 2.5. Tạo bảng DynamoDB

Năm bảng theo `MODEL_DESIGN_SPEC.md` mục 4.3. Dùng on-demand
(`PAY_PER_REQUEST`) để không phải đoán capacity và không bị tính tiền lúc không
dùng. Provisioned capacity tính tiền 24/7 kể cả khi không ai gọi; ở quy mô đồ án
on-demand gần như miễn phí.

```bash
aws dynamodb create-table --table-name Movies \
  --attribute-definitions AttributeName=movie_id,AttributeType=N \
  --key-schema AttributeName=movie_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST --region $AWS_REGION
```

```bash
aws dynamodb create-table --table-name Interactions \
  --attribute-definitions AttributeName=user_id,AttributeType=N AttributeName=sk,AttributeType=S \
  --key-schema AttributeName=user_id,KeyType=HASH AttributeName=sk,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST --region $AWS_REGION
```

```bash
aws dynamodb create-table --table-name PopularMovies \
  --attribute-definitions AttributeName=ranking_type,AttributeType=S AttributeName=genre,AttributeType=S \
  --key-schema AttributeName=ranking_type,KeyType=HASH AttributeName=genre,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST --region $AWS_REGION
```

Tương tự cho `Users` (khóa `user_id`) và `RecommendationCache` (khóa `user_id` +
`scenario`).

> [!NOTE]
> Sort key của `Interactions` là chuỗi `interaction_timestamp#movie_id`. Ở lệnh
> trên nó tên là `sk` vì tên thuộc tính DynamoDB không chứa được ký tự `#`.
> `scripts/export_interactions.py` đọc lại timestamp từ chính khóa này khi item
> không có thuộc tính `timestamp` riêng.

**Kiểm tra:**

```bash
aws dynamodb list-tables --region $AWS_REGION
```

**Ba điều backend phải khớp.** Nếu backend ghi event theo schema khác thì
`scripts/export_interactions.py` không đọc được, và retrain sẽ chạy trên dữ liệu
rỗng **mà không báo lỗi**:

1. Tên bảng phải là **`Interactions`**, không phải `movie-recommendation-activity`
2. Item phải có trường **`sk`** dạng `interaction_timestamp#movie_id`
3. Item phải có trường **`value`** — chứa watch progress, số sao, hoặc sentiment.
   Thiếu nó thì mọi event `rating` và `watch` bị loại

Backend cũng phải gửi tên trường là **`event_type`** (không phải
`interaction_type`) và chỉ dùng 8 giá trị hợp lệ.

### 2.6. IAM

**Người/máy chạy `aws_sync.py` và `train.py`** cần đọc/ghi bucket:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::movie-recommendation-fcaj/*"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::movie-recommendation-fcaj"
    }
  ]
}
```

**SageMaker execution role** cần thêm quyền chạy job. Làm trên Console dễ hơn
CLI:

```
IAM → Roles → Create role
→ AWS service → Use case: SageMaker → SageMaker - Execution
→ Tên: MovieRecSageMakerRole
```

Trust policy được tạo sẵn theo mẫu:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "sagemaker.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}
```

Gắn policy S3 ở trên (Add permissions → Create inline policy → JSON), cộng
`AmazonSageMakerFullAccess` (hoặc tối thiểu: `sagemaker:CreateProcessingJob`,
`sagemaker:DescribeProcessingJob`, `logs:CreateLogStream`, `logs:PutLogEvents`,
`ecr:GetAuthorizationToken`, `ecr:BatchGetImage`).

**Kiểm tra và khai báo ARN:**

```bash
aws iam get-role --role-name MovieRecSageMakerRole --query Role.Arn --output text
```

```bash
export MOVIE_REC_SAGEMAKER_ROLE=arn:aws:iam::<account-id>:role/MovieRecSageMakerRole
```

Hoặc điền vào `sagemaker.role_arn` trong `configs/aws.yaml` cho vĩnh viễn.
Launcher đọc theo thứ tự `--role-arn` → biến môi trường → file config, và **từ
chối chạy nếu cả ba đều rỗng** thay vì fail giữa chừng sau khi đã upload.

**EC2 instance role** (nếu re-train trên EC2): policy S3 ở trên +
`dynamodb:Scan` trên bảng `Interactions`.

**Backend role**: chỉ cần `s3:GetObject` trên `artifacts/*` và `serving/*`, cộng
`dynamodb:GetItem`/`BatchGetItem`/`PutItem`/`Query` trên các bảng. Backend
**không** cần quyền ghi artifact.

Nếu backend dùng Amazon Comprehend để phân loại sentiment bình luận
(`docs/interaction_events_api.md` mục 1.3), thêm:

```json
{
  "Effect": "Allow",
  "Action": ["comprehend:DetectSentiment", "comprehend:BatchDetectSentiment"],
  "Resource": "*"
}
```

---

## 3. Đưa data lên S3

Chạy pipeline ở local trước — bước này không đụng tới AWS:

```bash
python scripts/run_data_pipeline.py
```

```bash
python scripts/build_similar_movies.py
```

Cài AWS extras rồi đẩy lên:

```bash
pip install -r requirements-aws.txt
```

```bash
python scripts/aws_sync.py push --dry-run
```

```bash
python scripts/aws_sync.py push
```

Khối lượng thực đo **988,3 MB**:

| Thư mục local | Prefix S3 | Dung lượng | Số file |
|---|---|---:|---:|
| `data/processed` | `processed/` | 265,1 MB | 10 |
| `data/features` | `features/` | 254,4 MB | 4 |
| `data/splits` | `splits/` | 240,4 MB | 4 |
| `artifacts` | `artifacts/` | 184,2 MB | 20 |
| `data/serving` | `serving/` | 44,1 MB | 6 |
| `reports` | `reports/` | 0,1 MB | 27 |

Cặp thư mục nào ánh xạ tới prefix nào được khai báo trong `configs/aws.yaml`
khối `sync.pairs`, không phải tham số dòng lệnh — để không ai gõ nhầm lúc 2 giờ
sáng. Kiểm tra lại:

```bash
python scripts/aws_sync.py list
```

**CSV gốc của Kaggle không nằm trong `sync.pairs`**: 940 MB, không bao giờ đổi,
và không có gì phía sau bước làm sạch đọc tới nó. Nếu muốn lưu vết thì đẩy tay
một lần:

```bash
aws s3 sync data/movies_dataset_raw s3://movie-recommendation-fcaj/raw/
```

**Kiểm tra:**

```bash
aws s3 ls s3://movie-recommendation-fcaj/ --recursive --summarize --human-readable
```

### Nạp DynamoDB

`data/serving/movies_serving.jsonl` (45.430 dòng, 30,7 MB) và
`data/serving/popular_movies.jsonl` (21 dòng) là hai file đã đóng gói sẵn cho
DynamoDB. Nạp bằng `BatchWriteItem` 25 item mỗi lần — 45.430 item là khoảng
1.818 batch. Việc này thuộc phần backend; module model chỉ sinh ra file.

**Kiểm tra:**

```bash
aws dynamodb scan --table-name PopularMovies --select COUNT --region $AWS_REGION
```

Phải ra 21. Với `Movies` thì dùng `describe-table` xem `ItemCount` (cập nhật
chậm, khoảng 6 tiếng một lần) thay vì `scan` — scan toàn bảng 45.430 item chỉ để
đếm là tốn tiền đọc không cần thiết.

---

## 4. Train lần đầu

```bash
python train.py --version v1.0.0
```

```bash
python evaluate.py --sample-users 5000
```

Thời gian thực đo: dựng ma trận rồi train ALS hết **31,2 giây** trên ma trận
262.571 × 17.608 với 12.634.032 ô khác 0.

`evaluate.py` ghi kết quả ngược vào `manifest.json` của artifact
(`MODEL_DESIGN_SPEC.md` mục 14.2), nên **phải chạy evaluate trước khi đẩy lên
S3** — nếu không artifact sẽ mang `metrics: null`, và lần re-train sau không có
gì để so sánh, cổng kiểm duyệt sẽ tự động cho qua.

Kết quả tham chiếu trên tập test, 5.000 user:

| Model | HitRate@10 | NDCG@10 | So với baseline |
|---|---:|---:|---|
| `popularity_train` | 0,0332 | 0,0201 | — |
| `collaborative_als` | **0,1115** | **0,0537** | **+235,8%** |
| `hybrid_rrf` | 0,0818 | 0,0393 | +146,4% |
| `content_tfidf` | 0,0051 | 0,0035 | −84,8% |

`content_tfidf` thua baseline là đúng kỳ vọng: nó tồn tại để giải cold-start,
không phải để thắng trên user có lịch sử dày.

```bash
python scripts/aws_sync.py push --only artifacts reports
```

Kiểm tra `LATEST.json` đã trỏ đúng:

```bash
aws s3 cp s3://movie-recommendation-fcaj/artifacts/LATEST.json -
```

---

## 5. Re-train trên SageMaker Processing Job

### 5.1. Vì sao Processing Job chứ không phải Training Job

Ba lý do:

* việc cần làm không chỉ là fit — nó nạp event, dựng lại split, đánh giá và
  quyết định có thăng cấp hay không; Processing Job là hình dạng khớp với một
  script;
* artifact hệ thống phục vụ là cặp ma trận `.npy` đọc thẳng từ S3, không phải
  SageMaker Model, nên phần đóng gói model của Training Job không dùng tới;
* endpoint chạy liên tục là khoản duy nhất có thể vỡ ngân sách, và ở đây không
  có gì tạo ra nó.

Job chỉ tồn tại trong lúc chạy: bấm submit → AWS cấp máy → chạy khoảng 10 phút →
tự huỷ → ngừng tính tiền. Không có gì nằm lại.

### 5.2. Xem kế hoạch trước

```bash
python scripts/sagemaker_retrain_job.py --dry-run
```

Ba thứ phải kiểm tra trong output:

* `region` và `bucket` đúng
* `source_bundle.megabytes` phải là **0.46**, `files` là **59**
* `framework` là `sklearn 1.4-2`

Launcher dựng một thư mục staging chỉ chứa mã nguồn rồi mới upload. Trỏ
`source_dir` thẳng vào gốc repo sẽ nén cả `data/` và virtualenv — khoảng 2,2 GB
mỗi lần submit, mà job đã tự `--pull` dữ liệu từ S3 rồi. Thấy con số hàng nghìn
MB nghĩa là bản sửa chưa được áp dụng; dừng lại, đừng submit.

### 5.3. Smoke test trước, full sau

Lần đầu đừng chạy full. Job nhỏ để lộ lỗi môi trường sớm mà rẻ:

```bash
python scripts/sagemaker_retrain_job.py --version v1.1.0-smoke --instance-type ml.m5.large --wait
```

Dòng đầu tiên trong log là phiên bản Python của container, do
`deploy/sagemaker_retrain.py` in ra. Đây là điều duy nhất không xác minh được từ
máy local: base image ship interpreter nào là tuỳ phiên bản framework. Container
cài `requirements-container.txt` (lower bound) chứ không phải `requirements.txt`
(pin chính xác), vì bản pin đòi Python ≥3.11 và pin cứng biến chuyện lệch Python
thành job chết lúc `pip install`, vài phút sau khi submit.

### 5.4. Chạy thật

```bash
python scripts/sagemaker_retrain_job.py --version v1.1.0 --events s3://movie-recommendation-fcaj/events/ --wait
```

Job làm đúng những việc `retrain.py` làm ở local, theo thứ tự:

| Bước | Nội dung |
|---|---|
| 1 | Tải `processed/`, `features/`, `splits/`, `serving/`, `artifacts/` từ S3 |
| 2 | Đọc event JSONL, quy đổi thành dòng huấn luyện, ghép vào bảng interaction |
| 3 | Dựng lại split theo thời gian trên toàn bộ dữ liệu hợp nhất |
| 4 | Train ALS thành một version mới |
| 5 | Đánh giá so với baseline popularity và model đang phục vụ |
| 6 | Thăng cấp `LATEST.json`, hoặc giữ nguyên; đẩy tất cả lên S3 |

Thời gian tham chiếu đo local: train cộng đánh giá hết 4 phút 42 giây. Trên
SageMaker cộng thêm thời gian kéo 988 MB từ S3, tổng khoảng 10 phút.

Theo dõi:

```bash
aws sagemaker describe-processing-job --processing-job-name <job-name> --region $AWS_REGION --query ProcessingJobStatus
```

```bash
aws logs tail /aws/sagemaker/ProcessingJobs --follow --region $AWS_REGION
```

Kết quả đổ về:

```
s3://movie-recommendation-fcaj/artifacts/collaborative/<version>/
s3://movie-recommendation-fcaj/artifacts/LATEST.json
s3://movie-recommendation-fcaj/reports/validation/retrain_report.md
```

### 5.5. Cổng kiểm duyệt — phần quan trọng nhất

Re-train chạy theo lịch, nghĩa là **không ai xem lại từng lần chạy**. Không có
cổng thì một lần export phản hồi bị lỗi sẽ âm thầm thay thế model đang chạy tốt,
và dấu hiệu đầu tiên là gợi ý tệ đi.

Ba điều kiện trong `configs/aws.yaml` khối `retraining.promotion`, mỗi điều kiện
chỉ có quyền **chặn**:

| Điều kiện | Ý nghĩa |
|---|---|
| `enough_users` | Chấm dưới 1.000 user thì con số là nhiễu, không phải phép đo. |
| `beats_popularity` | Model cá nhân hóa không thắng nổi "ai cũng xem phim hot" thì không đáng phục vụ (`MODEL_DESIGN_SPEC.md` mục 15.2). |
| `no_regression` | Không được tụt quá 5% so với model đang phục vụ. |

Điều kiện thứ ba là **dung sai chứ không phải yêu cầu tăng**: train lại trên dữ
liệu đã dịch chuyển làm chỉ số nhảy vài phần trăm theo cả hai chiều, đòi hỏi
lần nào cũng phải tốt hơn thì sẽ không bao giờ thăng cấp được lần nào.

Không đạt thì `LATEST.json` giữ nguyên, artifact mới vẫn được lưu để xem xét, và
**job vẫn kết thúc thành công** — cổng chặn là cổng làm đúng việc của nó, không
phải sự cố. Kết quả nằm ở `reports/validation/retrain_report.md`.

> [!WARNING]
> Hai chỉ số chỉ so sánh được khi đo cùng một cách. Đó là lý do
> `sample_users` và `seed` nằm trong `configs/aws.yaml` chứ không phải truyền
> tay. Nếu vẫn truyền `--sample-users` (chỉ nên dùng để chạy thử), báo cáo sẽ in
> cảnh báo rằng phép so sánh không có giá trị.

### 5.6. Đặt lịch tự động

Đừng đặt lịch cho tới khi đã chạy tay thành công ít nhất một lần và đọc hiểu
`retrain_report.md`.

Cách rẻ nhất là **EventBridge Scheduler** gọi thẳng SageMaker, không cần máy nào
bật thường trực:

```
EventBridge → Schedules → Create schedule
→ Recurring, cron: 0 18 ? * SUN *      (Chủ nhật 18:00 UTC = 01:00 thứ Hai giờ VN)
→ Target: AWS SageMaker → CreateProcessingJob
→ Role: MovieRecSageMakerRole
```

Tần suất hợp lý ở giai đoạn đồ án là **một lần mỗi tuần**. Dày hơn không giúp gì
khi lượng event còn nhỏ: cổng thăng cấp sẽ chặn vì không đủ 1.000 user được
chấm, mà mỗi lần chạy vẫn tốn tiền.

---

## 6. Re-train trên EC2

Rẻ hơn SageMaker nếu instance đã có sẵn, và dễ debug hơn vì SSH vào xem được.
Nếu chỉ để retrain thì EC2 đắt hơn Processing Job — chỉ chọn nó khi cần chạy thứ
khác trên cùng máy.

### 6.1. Tạo instance

* AMI: Amazon Linux 2023 hoặc Ubuntu 22.04
* Instance type: **t3.large** là mức nhỏ nhất chạy xong. ALS giữ ma trận thưa
  262k × 17,6k cộng hai ma trận factor float32, đỉnh khoảng 6 GB. t3.xlarge thì
  thoải mái.
* Ổ đĩa: 20 GB trở lên (dataset + vài version artifact)
* Gắn **instance role** ở mục 2.6, không dùng access key.

### 6.2. Cài đặt

```bash
sudo bash deploy/ec2_bootstrap.sh https://github.com/<user>/<web-repo>.git
```

Script này cài Python 3.11, clone repo, tạo virtualenv, cài cả hai file
requirements, rồi cài systemd timer.

Nếu thư mục ML nằm ở vị trí khác trong repo web:

```bash
sudo PROJECT_SUBDIR=movie-recommendation-system BRANCH=main bash deploy/ec2_bootstrap.sh <repo-url>
```

### 6.3. Lịch chạy

`deploy/movie-rec-retrain.timer` đặt lịch **hàng tuần**, Chủ nhật 02:00, không
phải hàng ngày: ALS cần đủ phản hồi mới thì chỉ số mới nhúc nhích khỏi nhiễu, và
mỗi lần chạy tốn một instance-hour — lịch hàng ngày tiêu gấp bảy lần tiền để
thăng cấp gần như cùng một model.

```bash
systemctl list-timers movie-rec-retrain.timer
```

Chạy ngay một lần, không đợi lịch:

```bash
sudo systemctl start movie-rec-retrain.service
```

```bash
journalctl -u movie-rec-retrain.service -f
```

### 6.4. Tiết kiệm: dừng instance giữa các lần chạy

Instance chỉ cần sống vài phút mỗi tuần. `Persistent=true` trong timer khiến
systemd chạy bù ngay khi máy bật lại, nên có thể dùng EventBridge Scheduler bật
máy trước giờ chạy và tắt sau đó, hoặc đơn giản là bật tay khi cần.

---

## 7. Nối vào repo web

### 7.1. Bố cục hai repo

Repo web là repo riêng; thư mục này là **một phần trong đó**:

```
<web-repo>/
├── backend/                        # FastAPI
├── frontend/                       # React
└── movie-recommendation-system/    # ← repo này
    ├── configs/
    ├── src/
    ├── artifacts/
    ├── train.py  retrain.py  inference.py  evaluate.py
    └── deploy/
```

Có ba cách ghép, xếp theo mức nên dùng:

**a) Git submodule — nên dùng.** Hai repo giữ lịch sử riêng, repo web ghim đúng
một commit của model, nên biết chính xác bản web nào chạy bản model nào.

```bash
cd <web-repo>
git submodule add https://github.com/CaPPok/movie-recommendation-system.git movie-recommendation-system
git commit -m "chore: add ML module as submodule"
```

Người clone repo web phải nhớ:

```bash
git clone --recurse-submodules <web-repo-url>
```

Nâng model lên bản mới:

```bash
cd movie-recommendation-system && git pull origin main && cd ..
git add movie-recommendation-system && git commit -m "chore: bump ML module"
```

**b) Git subtree.** Người dùng repo web không cần biết gì về submodule, đổi lại
lịch sử hai repo trộn vào nhau.

**c) Copy thẳng vào repo web.** Là tình trạng hiện tại của thư mục
`movie-recommendation-main/movie-recommendation-system`. Chạy được ngay nhưng
hai bản sẽ lệch nhau, và sau vài tuần không ai biết bản nào mới hơn. Chỉ dùng
tạm.

> [!IMPORTANT]
> Dù chọn cách nào, **`data/` và `artifacts/` không được commit vào git**.
> Dataset 1,7 GB và artifact hàng trăm MB thuộc về S3. Repo chỉ chứa mã nguồn và
> config; `aws_sync.py pull` mang dữ liệu về khi cần.

### 7.2. Backend gọi model thế nào

Model chạy **trong tiến trình backend**, không phải một service riêng
(`MODEL_DESIGN_SPEC.md` mục 4.2). Engine nạp artifact một lần lúc khởi động rồi
giữ trong RAM.

```python
# backend/app/services/recommendation_provider.py
from pathlib import Path

from src.data.config import load_config
from src.recommenders.engine import RecommendationEngine, RecommendationRequestError
from src.recommenders.feedback import build_recommend_request

ML_ROOT = Path(__file__).resolve().parents[3] / "movie-recommendation-system"

# Một lần duy nhất lúc khởi động. Không dựng lại engine cho mỗi request:
# nó nạp ma trận TF-IDF và hai ma trận factor vào RAM.
_config = load_config(ML_ROOT / "configs/data_pipeline.yaml")
_model_config = load_config(ML_ROOT / "configs/model_serving.yaml")
_engine = RecommendationEngine(_config, _model_config)


def recommend(body: dict) -> dict:
    try:
        return _engine.recommend(body).to_dict()
    except RecommendationRequestError as error:
        raise HTTPException(status_code=400, detail=str(error))
```

Để `import src...` hoạt động, thêm thư mục ML vào `sys.path` lúc khởi động, hoặc
đặt `PYTHONPATH=/app/movie-recommendation-system`.

**Artifact lấy từ đâu lúc backend khởi động:** container tải về từ S3 trước khi
uvicorn chạy.

```bash
python movie-recommendation-system/scripts/aws_sync.py pull --only artifacts serving
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`LATEST.json` được đọc **đúng một lần lúc khởi động**. Không đọc S3 lại trong
mỗi request — làm vậy là thêm một lần gọi mạng cho mỗi lần tải trang, đúng thứ
kiến trúc batch-first sinh ra để tránh (`MODEL_DESIGN_SPEC.md` mục 14.3).

### 7.3. Ba việc backend phải sửa

Theo `MODEL_DESIGN_SPEC.md` mục 16.5–16.7, backend hiện chưa khớp contract:

| Vấn đề | Hiện tại | Phải thành |
|---|---|---|
| Endpoint | `GET /recommend/{user_id}` | `POST /model/recommend`, vì `recent_interactions` không truyền qua query string được |
| Response | Trả cả `title` | Model **chỉ** trả `movie_id`, `score`, `reason_code`, `reason_context`; backend tự `BatchGetItem` bảng `Movies` lấy metadata |
| `event_type` | chuỗi tự do, không validate | chặn giá trị ngoài tám enum ở `docs/interaction_events_api.md` |

> [!WARNING]
> Guest phải gửi `user_id: null`, **không phải chuỗi `"guest"`**. Engine ép kiểu
> integer và sẽ trả 4xx nếu nhận được chuỗi. Danh sách 8 event type nên lấy động
> từ `score_interaction_events()` thay vì hardcode, để không lệch khi model đổi.

### 7.4. Frontend

Frontend chưa có mã tracking nào. Không có tracking thì scenario returning user
không có dữ liệu và vòng lặp phản hồi không demo được. Cần bắn event khi:

| Hành động | Event |
|---|---|
| Mở trang chi tiết phim | `click` |
| Xem qua ngưỡng 50% | `watch` (kèm `value` = tỉ lệ đã xem) |
| Xem hết | `complete` |
| Bấm thích / không thích | `like` / `dislike` |
| Chấm sao | `rating` (kèm `value` = số sao) |
| Bấm chia sẻ | `share` |
| Gửi bình luận | `comment` (kèm `value` = sentiment, xem bên dưới) |

`reason_code` trong response dùng để dựng câu giải thích. Model chỉ trả
`source_movie_id`; **backend lấy `title` từ bảng `Movies`**, không có placeholder
nào tự điền.

---

## 8. Vòng lặp phản hồi đầy đủ

```
Người dùng share / comment / xem phim
        ↓
Frontend bắn event  ──►  Backend validate 8 enum
        ↓
DynamoDB  Interactions
        ↓  scripts/export_interactions.py --upload   (hằng tuần)
S3  events/2026-07-27.jsonl
        ↓  retrain.py --events s3://.../events/
Quy đổi điểm tương tác  →  ghép vào bảng interaction  →  dựng lại split
        ↓
Train ALS  →  Đánh giá  →  Cổng kiểm duyệt
        ↓ (đạt)
S3  artifacts/collaborative/v1.1.0/  +  LATEST.json
        ↓  khởi động lại backend
Gợi ý mới
```

### 8.1. Điểm tương tác biến thành dữ liệu huấn luyện thế nào

Chi tiết đầy đủ ở `docs/interaction_events_api.md`. Tóm tắt phần liên quan tới
huấn luyện:

Điểm tổng hợp của mỗi cặp (user, phim) được chiếu lên thang rating 0,5–5,0 qua
ba mốc, để `build_training_matrix` dùng lại đúng ngưỡng cũ mà không phải học
thêm thang thứ hai:

| Tương tác | Điểm | Rating quy đổi | Vào model? |
|---|---:|---:|---|
| `share` hoặc `like` | +15 | 5,00 | có, confidence cao nhất |
| `comment` sentiment +1 | +15 | 5,00 | có |
| `complete` | +12 | 4,60 | có |
| `watch` ≥ 50% | +10 | 4,33 | có |
| `comment` sentiment 0 | +3 | 3,40 | không (vùng trung tính) |
| `click` | +2 | 3,27 | không |
| `comment` sentiment −1 | −9 | 1,50 | không, và bị lọc khỏi gợi ý |
| `dislike` | −15 | 0,50 | không, và bị lọc khỏi gợi ý |
| **`comment` không có sentiment** | — | — | **không, event bị bỏ hẳn** |

Ngưỡng: `>= 4,0` là positive, `<= 2,5` là negative, khoảng giữa bị loại khỏi ma
trận huấn luyện theo `neutral_rating_policy` (`MODEL_DESIGN_SPEC.md` mục 7.3).

Một điểm cần biết: `rating` 5 sao gửi qua đường event sinh ra +10 điểm, quy đổi
lại thành 4,33 chứ không phải 5,0 — hai thang không khớp tuyệt đối. Điều này
không ảnh hưởng dữ liệu lịch sử, vì rating gốc từ dataset đã nằm trong bảng
interaction dưới dạng giá trị thật, và chính sách `prefer_rating` giữ nguyên
chúng.

Ba quyết định đi kèm, tất cả để bản train lại còn so sánh được với bản nó thay
thế:

* **Tắt suy giảm theo thời gian khi tạo dữ liệu huấn luyện.** Lúc phục vụ, suy
  giảm trả lời "hôm nay người này muốn gì". Lúc huấn luyện thì cần toàn bộ lịch
  sử được cân như nhau ở mọi lần chạy. Để bật, hai lần chạy trên cùng dữ liệu sẽ
  ra hai tập huấn luyện khác nhau và không quy được sự thay đổi chỉ số về đâu.
* **Rating tường minh thắng.** Người vừa chấm sao vừa tương tác sinh ra hai dòng
  ứng viên; giữ dòng rating.
* **Dòng suy ra mang `interaction_type = "event"`**, tách bạch với `rating` gốc
  từ dataset Kaggle, đúng yêu cầu `MODEL_DESIGN_SPEC.md` mục 11.3.

### 8.2. Bao lâu re-train một lần

Hàng tuần. Trước đó phải có **đủ event mới** — vài trăm event trên 26 triệu
tương tác lịch sử thì không dịch chuyển được ALS. Xem
`reports/validation/retrain_report.md` phần "Sự kiện đưa vào huấn luyện": nếu
"Dòng huấn luyện sinh ra" chỉ vài trăm, model mới về cơ bản là model cũ và cổng
kiểm duyệt sẽ quyết định dựa trên nhiễu.

---

## 9. Chi phí và cách kiểm soát

Con số dưới đây là **bậc độ lớn để ra quyết định**, không phải báo giá. Giá thay
đổi theo thời điểm và region; kiểm tra lại trên trang pricing của AWS trước khi
cam kết ngân sách.

### 9.1. Cái gì tốn tiền

| Hạng mục | Khối lượng thật | Cách tính tiền | Bậc độ lớn |
|---|---|---|---|
| S3 lưu trữ | 988 MB đẩy lên, ~1 GB lưu | theo GB-tháng | vài cent/tháng |
| S3 request | ~70 object mỗi lần sync | theo 1.000 request | không đáng kể |
| SageMaker Processing Job `ml.m5.xlarge` | ~10 phút/lần chạy | **theo giây, chỉ khi job chạy** | ~0,04 USD/lần |
| DynamoDB `Movies` | 45.430 item nạp một lần | on-demand, theo request | dưới 0,10 USD |
| DynamoDB `PopularMovies` | 21 item | on-demand | không đáng kể |
| DynamoDB `Interactions` | theo lượng event thật | on-demand | phụ thuộc traffic |
| EC2 `t3.large` 30 phút/tuần | tuỳ chọn thay SageMaker | theo giờ | ~0,05 USD/tuần |
| CloudWatch Logs | log mỗi lần job chạy | theo GB nạp vào | không đáng kể |
| **SageMaker real-time endpoint** | — | **24/7 kể từ lúc tạo** | **~50 USD/tháng — thiết kế này không dùng** |

Ở quy mô đồ án, tổng chi phí phần ML nằm ở mức **vài chục cent tới vài USD mỗi
tháng**, miễn là tránh được ba khoản ở mục sau.

Chỉ precompute recommendation cho **một tập user demo** (đề xuất 1.000 user),
không precompute cho toàn bộ 270.883 user.

### 9.2. Ba khoản có thể phá ngân sách

**SageMaker real-time endpoint — nguy hiểm nhất.** Endpoint tính tiền 24/7 kể từ
lúc tạo, kể cả khi không ai gọi. Một `ml.m5.xlarge` chạy cả tháng đắt hơn toàn bộ
phần còn lại của hệ thống cộng lại.

Kiến trúc này cố ý không tạo endpoint nào. Nhưng template web có file
`ml/sagemaker/deploy_model.py` tạo endpoint thật với image xgboost. **Đừng chạy
nó.** Xoá endpoint chưa đủ, phải xoá cả endpoint-config và model.

**S3 versioning phình vô hạn.** Xem mục 2.4 — lifecycle rule là bắt buộc, không
phải tuỳ chọn.

**EC2 quên tắt.** Instance bật 24/7 chỉ để chạy 30 phút mỗi tuần thì đắt hơn
Processing Job nhiều lần. Xem mục 6.4.

### 9.3. Kiểm tra hằng tuần

Không có endpoint nào — kiểm tra quan trọng nhất, kết quả phải rỗng:

```bash
aws sagemaker list-endpoints --region $AWS_REGION
```

Có endpoint lạ thì xoá ngay:

```bash
aws sagemaker delete-endpoint --endpoint-name <ten> --region $AWS_REGION
```

```bash
aws sagemaker delete-endpoint-config --endpoint-config-name <ten> --region $AWS_REGION
```

Không có EC2 quên tắt:

```bash
aws ec2 describe-instances --region $AWS_REGION --filters Name=instance-state-name,Values=running --query "Reservations[].Instances[].[InstanceId,InstanceType]" --output table
```

Bucket không phình:

```bash
aws s3 ls s3://movie-recommendation-fcaj/ --recursive --summarize --human-readable
```

`Total Size` nên quanh 1 GB. Vượt 3 GB nghĩa là lifecycle rule chưa chạy hoặc
chưa được tạo.

### 9.4. Khi hoá đơn tăng bất thường

Mở Cost Explorer, nhóm theo **Service** rồi theo **Usage Type**. Ba nghi phạm
theo thứ tự khả năng: SageMaker endpoint ai đó lỡ tạo, EC2 quên tắt, S3 version
cũ tích luỹ.

---

## 10. Vận hành, giám sát và quay lui

### Quay lui model

Sửa `LATEST.json` trỏ về version cũ rồi **khởi động lại** backend. Không có cơ
chế nạp nóng trong MVP (`MODEL_DESIGN_SPEC.md` mục 14.3).

```bash
aws s3 cp s3://movie-recommendation-fcaj/artifacts/LATEST.json ./LATEST.json
```

Sửa `"collaborative": "v1.0.0"`, rồi:

```bash
aws s3 cp ./LATEST.json s3://movie-recommendation-fcaj/artifacts/LATEST.json
```

> [!NOTE]
> Cơ chế này **không** cho phép "đổi model mà không deploy code" trong trường hợp
> tổng quát. Nó chỉ chạy khi model mới có cùng định dạng artifact và cùng logic
> nạp. Đổi thuật toán hay đổi schema artifact thì vẫn phải sửa và triển khai lại
> mã nguồn.

### Không bao giờ ghi đè artifact cũ

Mỗi lần train sinh một thư mục version mới. Bucket đã bật versioning, nhưng đó
là lưới an toàn chứ không phải quy trình.

### Cần xem gì khi gợi ý xuống chất lượng

| Nguồn | Xem cái gì |
|---|---|
| `reports/validation/retrain_report.md` | Lần chạy gần nhất có thăng cấp không, và vì sao |
| `reports/validation/retrain_history/` | Chỉ số theo từng version, để biết chất lượng tụt từ lúc nào |
| `reports/validation/model_evaluation.md` | So sánh đầy đủ bốn model |
| Log backend | `fallback_level` khác `none` nhiều nghĩa là nguồn cá nhân hóa đang rỗng |
| Log backend | `events_ignored` > 0 nghĩa là frontend bắn event model chưa biết |

---

## 11. Dọn dẹp khi kết thúc

Xoá theo thứ tự này để không sót gì tính tiền.

```bash
aws sagemaker list-endpoints --region $AWS_REGION
```

Có endpoint thì xoá endpoint, endpoint-config và model trước.

```bash
aws s3 rm s3://movie-recommendation-fcaj --recursive
```

> [!WARNING]
> Bucket bật versioning thì lệnh trên **không xoá version cũ**, và bucket sẽ
> không xoá được trong khi vẫn tính tiền lưu trữ. Phải làm qua Console
> (`S3 → bucket → Empty`) hoặc xoá từng version bằng API.

```bash
aws s3api delete-bucket --bucket movie-recommendation-fcaj --region $AWS_REGION
```

```bash
aws dynamodb delete-table --table-name Movies --region $AWS_REGION
```

Lặp lại cho `PopularMovies`, `Interactions`, `Users`, `RecommendationCache`.

Cuối cùng xoá IAM role và kiểm tra Billing sau 2 ngày để chắc chắn không còn
khoản nào phát sinh.

---

## 12. Checklist triển khai

**Trước khi tạo gì**

- [ ] Quota `ml.m5.xlarge for processing job usage` ≥ 1
- [ ] Budget alarm đã bật, có email nhận cảnh báo
- [ ] Cost Explorer đã bật
- [ ] `aws sts get-caller-identity` chạy được

**Một lần duy nhất**

- [ ] Chốt region, ghi vào `configs/aws.yaml` và biến môi trường mọi nơi
- [ ] Tạo S3 bucket, chặn public access, bật versioning
- [ ] **Lifecycle rule xoá version cũ sau 30 ngày**
- [ ] Tạo 5 bảng DynamoDB, tất cả on-demand
- [ ] Tạo IAM role: sync/train, SageMaker, EC2, backend
- [ ] `python scripts/run_data_pipeline.py`
- [ ] `python scripts/build_similar_movies.py`
- [ ] `python scripts/aws_sync.py push`
- [ ] Nạp `Movies` (45.430 item) và `PopularMovies` (21 item) vào DynamoDB
- [ ] `python train.py --version v1.0.0`
- [ ] `python evaluate.py --sample-users 5000` ← **trước khi push artifact**
- [ ] `python scripts/aws_sync.py push --only artifacts reports`
- [ ] `--dry-run` cho `source_bundle` = 0.46 MB / 59 file
- [ ] Smoke test `ml.m5.large`, ghi lại phiên bản Python container
- [ ] Thêm repo này làm submodule của repo web
- [ ] Backend: đổi sang `POST /model/recommend`, bỏ `title` khỏi response
- [ ] Backend: validate `event_type` theo 8 enum, guest gửi `user_id: null`
- [ ] Backend: bảng `Interactions` có `sk` và `value`
- [ ] Frontend: thêm tracking 8 event, gồm `share` và `comment`

**Định kỳ**

- [ ] `python scripts/export_interactions.py --upload`
- [ ] `python scripts/sagemaker_retrain_job.py --version vX.Y.Z --events s3://.../events/` (hoặc để EventBridge / EC2 timer tự chạy)
- [ ] Đọc `reports/validation/retrain_report.md`
- [ ] Nếu thăng cấp: khởi động lại backend để nạp artifact mới
- [ ] `aws sagemaker list-endpoints` phải rỗng
- [ ] Kiểm tra dung lượng bucket quanh 1 GB

**Kiểm tra không cần AWS**

```bash
python -m pytest -q
```

```bash
python scripts/score_interactions.py --demo
```

```bash
python inference.py --demo
```

```bash
python retrain.py --dry-run
```

```bash
python scripts/sagemaker_retrain_job.py --dry-run
```
