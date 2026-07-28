# Runbook dựng hạ tầng AWS

Làm theo thứ tự, mỗi bước có lệnh kiểm tra. Giả định frontend và backend đã chạy trên AWS; tài liệu này chỉ dựng phần hạ tầng cho `movie-recommendation-system`.

**Quan hệ với [`aws_deployment.md`](aws_deployment.md):** tài liệu kia giải thích *vì sao* mỗi lựa chọn kiến trúc lại như vậy và backend phải sửa gì. Tài liệu này là *làm gì, theo thứ tự nào, kiểm tra ra sao*, cộng phần kiểm soát chi phí. Cần hiểu lý do thì đọc file kia; cần dựng cho xong thì đọc file này.

Mọi số đo trong tài liệu lấy từ lần chạy đầy đủ ngày 2026-07-28 trên máy local.

Region cố định: **`ap-southeast-1`** (Singapore). Bucket: **`movie-recommendation-fcaj`**.

---

## Mục lục

0. [Hai việc làm trước tiên](#0-hai-việc-làm-trước-tiên)
1. [Chi phí — đọc trước khi tạo gì](#1-chi-phí--đọc-trước-khi-tạo-gì)
2. [Credentials](#2-credentials)
3. [S3](#3-s3)
4. [IAM](#4-iam)
5. [DynamoDB](#5-dynamodb)
6. [Nạp dữ liệu](#6-nạp-dữ-liệu)
7. [SageMaker Processing Job](#7-sagemaker-processing-job)
8. [Lịch retrain tự động](#8-lịch-retrain-tự-động)
9. [Giám sát chi phí](#9-giám-sát-chi-phí)
10. [Dọn dẹp](#10-dọn-dẹp)
11. [Checklist](#11-checklist)

---

## 0. Hai việc làm trước tiên

Hai việc này chờ lâu hoặc phòng ngừa thiệt hại, nên làm trước mọi thứ khác.

### 0.1. Xin quota SageMaker

Tài khoản AWS mới thường có quota **0** cho instance SageMaker. Phát hiện chuyện này lúc sắp chạy job là mất vài ngày chờ.

```
Service Quotas → AWS services → Amazon SageMaker
→ tìm "ml.m5.xlarge for processing job usage"
```

Bằng 0 thì bấm **Request increase**, xin 2. Duyệt mất vài giờ tới vài ngày. Xin luôn `ml.m5.large for processing job usage` để chạy smoke test rẻ hơn.

### 0.2. Dựng hàng rào ngân sách

Làm **trước** khi tạo tài nguyên, để nếu có gì chạy sai thì biết trong vòng một ngày chứ không phải cuối tháng.

```
Billing and Cost Management → Budgets → Create budget
→ Cost budget → Monthly → đặt ngưỡng (ví dụ 10 USD)
→ Alert khi actual > 50% và > 80%, gửi về email của bạn
```

Bật thêm biểu đồ chi phí theo dịch vụ:

```
Billing → Cost Explorer → Enable
```

Cost Explorer mất 24 giờ để có dữ liệu đầu tiên, nên bật sớm.

---

## 1. Chi phí — đọc trước khi tạo gì

Con số dưới đây là **bậc độ lớn để ra quyết định**, không phải báo giá. Giá thay đổi theo thời điểm và region; kiểm tra lại trên trang pricing của AWS trước khi cam kết ngân sách.

### Cái gì tốn tiền

| Hạng mục | Khối lượng thật | Cách tính tiền | Bậc độ lớn |
|---|---|---|---|
| S3 lưu trữ | 988 MB đẩy lên, ~1 GB lưu | theo GB-tháng | vài cent/tháng |
| S3 request | ~70 object mỗi lần sync | theo 1.000 request | không đáng kể |
| SageMaker Processing Job | ~10 phút/lần chạy | **theo giây, chỉ khi job chạy** | vài cent/lần |
| DynamoDB `Movies` | 45.430 item nạp một lần | on-demand, theo request | dưới 0,10 USD |
| DynamoDB `PopularMovies` | 21 item | on-demand | không đáng kể |
| DynamoDB `Interactions` | theo lượng event thật | on-demand | phụ thuộc traffic |
| CloudWatch Logs | log mỗi lần job chạy | theo GB nạp vào | không đáng kể |

Ở quy mô đồ án, tổng chi phí phần ML nằm ở mức **vài chục cent tới vài USD mỗi tháng**, miễn là tránh được ba khoản dưới đây.

### Ba khoản có thể phá ngân sách

**SageMaker real-time endpoint — nguy hiểm nhất.** Endpoint tính tiền **24/7 kể từ lúc tạo**, kể cả khi không ai gọi. Một `ml.m5.xlarge` chạy cả tháng đắt hơn toàn bộ phần còn lại của hệ thống cộng lại, cỡ vài chục USD/tháng.

Kiến trúc này **cố ý không tạo endpoint nào**. Nhưng template web có file `ml/sagemaker/deploy_model.py` tạo endpoint thật. Đừng chạy nó. Kiểm tra định kỳ:

```bash
aws sagemaker list-endpoints --region ap-southeast-1
```

Kết quả phải là danh sách rỗng. Có endpoint lạ thì xoá ngay:

```bash
aws sagemaker delete-endpoint --endpoint-name <ten> --region ap-southeast-1
```

Xoá endpoint chưa đủ, phải xoá cả endpoint-config và model:

```bash
aws sagemaker delete-endpoint-config --endpoint-config-name <ten> --region ap-southeast-1
```

**S3 versioning phình vô hạn.** Bật versioning là đúng (retrain đẩy nhầm artifact hỏng còn quay lại được), nhưng mỗi lần `aws_sync.py push` ghi đè sẽ tạo một version mới của **mọi file**, và version cũ vẫn tính tiền lưu trữ. Sau 10 lần retrain là 10 bản của 988 MB. Lifecycle rule ở [mục 3.3](#33-lifecycle-rule--bắt-buộc-nếu-bật-versioning) xử lý việc này.

**EC2 quên tắt.** Nếu chọn retrain trên EC2 thay vì SageMaker, instance tính tiền theo giờ kể cả lúc rảnh. Repo có sẵn systemd timer chạy định kỳ; nhưng nếu instance bật 24/7 chỉ để chạy 30 phút mỗi tuần thì đắt hơn Processing Job nhiều lần. Xem `aws_deployment.md` mục 6.4.

### Vì sao Processing Job rẻ

Job chỉ tồn tại trong lúc chạy. Bấm submit → AWS cấp máy → chạy ~10 phút → tự huỷ → ngừng tính tiền. Không có gì nằm lại. Đây là lý do toàn bộ kiến trúc là batch-first: mọi thứ nặng tính sẵn thành file trên S3, lúc phục vụ chỉ đọc file, không cần máy nào bật thường trực cho ML.

---

## 2. Credentials

```bash
aws --version
```

Chưa có thì cài AWS CLI v2. Sau đó cấu hình — **tự bạn nhập key**, đừng dán vào file trong repo hay vào chat:

```bash
aws configure
```

Region `ap-southeast-1`, output `json`.

**Kiểm tra:**

```bash
aws sts get-caller-identity
```

Ghi lại `Account` trong kết quả, các bước sau cần account id.

---

## 3. S3

### 3.1. Tạo bucket

Tên bucket là duy nhất toàn cầu. Nếu `movie-recommendation-fcaj` đã bị người khác lấy, đổi tên rồi sửa `aws.bucket` trong `configs/aws.yaml`.

```bash
aws s3api create-bucket --bucket movie-recommendation-fcaj --region ap-southeast-1 --create-bucket-configuration LocationConstraint=ap-southeast-1
```

### 3.2. Chặn truy cập công khai

```bash
aws s3api put-public-access-block --bucket movie-recommendation-fcaj --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

### 3.3. Versioning và lifecycle rule

```bash
aws s3api put-bucket-versioning --bucket movie-recommendation-fcaj --versioning-configuration Status=Enabled
```

Versioning không có lifecycle sẽ tích luỹ vô hạn. Tạo file `lifecycle.json`:

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
aws s3api put-bucket-lifecycle-configuration --bucket movie-recommendation-fcaj --lifecycle-configuration file://lifecycle.json
```

Version cũ tự xoá sau 30 ngày — vẫn đủ dài để quay lui model, đủ ngắn để không tích tiền. Rule thứ hai dọn phần upload dở dang: upload 988 MB đứt giữa chừng để lại mảnh vụn vẫn tính tiền mà không nhìn thấy trong danh sách file.

**Kiểm tra:**

```bash
aws s3api get-bucket-lifecycle-configuration --bucket movie-recommendation-fcaj
```

---

## 4. IAM

Ba vai trò với ba mức quyền khác nhau. Nguyên tắc: backend **không bao giờ** được quyền ghi artifact.

### 4.1. SageMaker execution role

Làm trên Console dễ hơn CLI:

```
IAM → Roles → Create role
→ AWS service → Use case: SageMaker → SageMaker - Execution
→ Tên: MovieRecSageMakerRole
```

Policy `AmazonSageMakerFullAccess` được gắn sẵn. Thêm quyền S3 giới hạn đúng bucket qua **Add permissions → Create inline policy → JSON**:

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

**Kiểm tra:**

```bash
aws iam get-role --role-name MovieRecSageMakerRole --query Role.Arn --output text
```

Khai báo ARN cho launcher. Trong PowerShell:

```bash
$env:MOVIE_REC_SAGEMAKER_ROLE = "arn:aws:iam::<account-id>:role/MovieRecSageMakerRole"
```

Hoặc điền vào `sagemaker.role_arn` trong `configs/aws.yaml` cho vĩnh viễn. Launcher đọc theo thứ tự `--role-arn` → biến môi trường → file config, và từ chối chạy nếu cả ba đều rỗng.

### 4.2. Backend role — chỉ đọc

Backend chỉ cần đọc artifact và bảng tra cứu:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": [
        "arn:aws:s3:::movie-recommendation-fcaj/artifacts/*",
        "arn:aws:s3:::movie-recommendation-fcaj/serving/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["dynamodb:GetItem", "dynamodb:BatchGetItem", "dynamodb:Query", "dynamodb:PutItem"],
      "Resource": "arn:aws:dynamodb:ap-southeast-1:<account-id>:table/*"
    }
  ]
}
```

Nếu backend dùng Amazon Comprehend để chấm sentiment bình luận, thêm `comprehend:DetectSentiment` và `comprehend:BatchDetectSentiment`. Không có sentiment thì event `comment` bị bỏ qua hoàn toàn — đó là hành vi cố ý, không phải lỗi.

### 4.3. Người chạy sync

Tài khoản IAM bạn dùng ở máy local cần đúng policy S3 ở mục 4.1, cộng `sagemaker:CreateProcessingJob` và `sagemaker:DescribeProcessingJob` để submit job.

---

## 5. DynamoDB

Năm bảng, tất cả dùng **on-demand** (`PAY_PER_REQUEST`). Lý do liên quan trực tiếp tới chi phí: provisioned capacity tính tiền cả lúc không ai dùng, còn on-demand chỉ tính theo request thật. Ở quy mô đồ án, on-demand gần như miễn phí; provisioned thì tính tiền 24/7.

```bash
aws dynamodb create-table --table-name Movies --attribute-definitions AttributeName=movie_id,AttributeType=N --key-schema AttributeName=movie_id,KeyType=HASH --billing-mode PAY_PER_REQUEST --region ap-southeast-1
```

```bash
aws dynamodb create-table --table-name PopularMovies --attribute-definitions AttributeName=ranking_type,AttributeType=S AttributeName=genre,AttributeType=S --key-schema AttributeName=ranking_type,KeyType=HASH AttributeName=genre,KeyType=RANGE --billing-mode PAY_PER_REQUEST --region ap-southeast-1
```

```bash
aws dynamodb create-table --table-name Interactions --attribute-definitions AttributeName=user_id,AttributeType=N AttributeName=sk,AttributeType=S --key-schema AttributeName=user_id,KeyType=HASH AttributeName=sk,KeyType=RANGE --billing-mode PAY_PER_REQUEST --region ap-southeast-1
```

Tương tự cho `Users` (khoá `user_id`) và `RecommendationCache` (khoá `user_id` + `scenario`).

Sort key của `Interactions` là chuỗi `interaction_timestamp#movie_id`, nhưng đặt tên thuộc tính là `sk` vì tên thuộc tính DynamoDB không chứa được ký tự `#`.

**Kiểm tra:**

```bash
aws dynamodb list-tables --region ap-southeast-1
```

### Ba điều backend phải khớp

Nếu backend đang chạy với schema khác thì `scripts/export_interactions.py` sẽ không đọc được event, và retrain sẽ chạy trên dữ liệu rỗng mà không báo lỗi:

1. Tên bảng phải là **`Interactions`**, không phải `movie-recommendation-activity`
2. Item phải có trường **`sk`** dạng `interaction_timestamp#movie_id`
3. Item phải có trường **`value`** — chứa watch progress, số sao, hoặc sentiment. Thiếu nó thì mọi event `rating` và `watch` bị loại

Backend cũng phải gửi tên trường là **`event_type`** (không phải `interaction_type`) và chỉ dùng 8 giá trị hợp lệ. Danh sách 8 loại lấy động từ `score_interaction_events()` thay vì hardcode, để không lệch khi model đổi.

---

## 6. Nạp dữ liệu

### 6.1. Đẩy lên S3

```bash
python scripts/aws_sync.py push --dry-run
```

Xem kỹ danh sách rồi chạy thật:

```bash
python scripts/aws_sync.py push
```

Khối lượng thực đo **988,3 MB**:

| Thư mục local | Prefix S3 | Dung lượng |
|---|---|---:|
| `data/processed` | `processed/` | 265,1 MB |
| `data/features` | `features/` | 254,4 MB |
| `data/splits` | `splits/` | 240,4 MB |
| `artifacts` | `artifacts/` | 184,2 MB |
| `data/serving` | `serving/` | 44,1 MB |
| `reports` | `reports/` | 0,1 MB |

`data/movies_dataset_raw/` **không** nằm trong `sync.pairs`, nên 940 MB CSV gốc không bị upload. Job retrain đọc `data/processed`, không cần CSV thô.

**Kiểm tra:**

```bash
aws s3 ls s3://movie-recommendation-fcaj/ --recursive --summarize --human-readable
```

### 6.2. Nạp DynamoDB

Hai file đã đóng gói sẵn cho DynamoDB:

- `data/serving/movies_serving.jsonl` — 45.430 dòng, 30,7 MB → bảng `Movies`
- `data/serving/popular_movies.jsonl` — 21 dòng → bảng `PopularMovies`

Nạp bằng `BatchWriteItem` 25 item mỗi lần. 45.430 item chia thành khoảng 1.818 batch.

**Kiểm tra:**

```bash
aws dynamodb scan --table-name PopularMovies --select COUNT --region ap-southeast-1
```

Phải ra 21. Với `Movies` thì dùng `describe-table` xem `ItemCount` (cập nhật chậm, khoảng 6 tiếng một lần) thay vì `scan` — scan toàn bảng 45.430 item là tốn tiền đọc không cần thiết.

---

## 7. SageMaker Processing Job

### 7.1. Xem kế hoạch trước

```bash
python scripts/sagemaker_retrain_job.py --dry-run
```

Ba thứ phải kiểm tra trong output:

- `region` và `bucket` đúng
- `source_bundle.megabytes` phải là **0.46**, `files` là **59**. Thấy con số hàng nghìn MB nghĩa là đang gói cả `data/` và virtualenv — dừng lại, đừng submit.
- `framework` là `sklearn 1.4-2`

### 7.2. Smoke test trước, full sau

Lần đầu đừng chạy full. Job nhỏ để lộ lỗi môi trường sớm mà rẻ:

```bash
python scripts/sagemaker_retrain_job.py --version v1.1.0-smoke --instance-type ml.m5.large --wait
```

Dòng đầu tiên trong log là phiên bản Python của container, do `deploy/sagemaker_retrain.py` in ra. Đây là điều duy nhất không xác minh được từ máy local: base image ship interpreter nào là tuỳ phiên bản framework. `requirements-container.txt` dùng lower bound nên cài được trên Python 3.9 trở lên; nếu vẫn lỗi lúc `pip install` thì con số đó là manh mối đầu tiên.

### 7.3. Chạy thật

```bash
python scripts/sagemaker_retrain_job.py --version v1.1.0 --wait
```

Có event thật đã export từ DynamoDB thì thêm `--events s3://movie-recommendation-fcaj/events/2026-07-28/`. Không có `--events` thì job vẫn train lại trên dữ liệu hiện có — hữu ích để xác nhận đường ống, nhưng model gần như y hệt.

Job chạy sáu bước: pull từ S3 → nạp event → dựng lại split → train → đánh giá → thăng cấp hoặc giữ nguyên. Thời gian tham chiếu đo trên máy local: train ALS 31 giây, train cộng đánh giá 4 phút 42 giây. Trên SageMaker cộng thêm thời gian kéo 988 MB từ S3, tổng khoảng 10 phút.

### 7.4. Theo dõi

```bash
aws sagemaker list-processing-jobs --region ap-southeast-1 --max-results 5
```

```bash
aws sagemaker describe-processing-job --processing-job-name <job-name> --region ap-southeast-1 --query ProcessingJobStatus
```

```bash
aws logs tail /aws/sagemaker/ProcessingJobs --follow --region ap-southeast-1
```

### 7.5. Kết quả

```
s3://movie-recommendation-fcaj/artifacts/collaborative/<version>/
s3://movie-recommendation-fcaj/artifacts/LATEST.json
s3://movie-recommendation-fcaj/reports/validation/retrain_report.md
```

`LATEST.json` chỉ dịch chuyển nếu ứng viên qua cả ba cổng: đủ 1.000 user được chấm, vượt baseline popularity trong cùng lần chạy, và không tụt quá 5% so với model đang phục vụ. Không qua thì artifact vẫn được giữ để soi, con trỏ đứng yên. Đọc `retrain_report.md` để biết cổng nào chặn.

Backend phải **khởi động lại** để nạp artifact mới — MVP không có cơ chế nạp nóng.

---

## 8. Lịch retrain tự động

Đừng đặt lịch cho tới khi đã chạy tay thành công ít nhất một lần và đọc hiểu `retrain_report.md`.

Cách rẻ nhất là **EventBridge Scheduler** gọi thẳng SageMaker, không cần máy nào bật thường trực:

```
EventBridge → Schedules → Create schedule
→ Recurring, cron: 0 18 ? * SUN *      (Chủ nhật 18:00 UTC = 01:00 thứ Hai giờ VN)
→ Target: AWS SageMaker → CreateProcessingJob
→ Role: MovieRecSageMakerRole
```

Tần suất hợp lý ở giai đoạn đồ án là **một lần mỗi tuần**. Retrain dày hơn không giúp gì khi lượng event còn nhỏ: cổng thăng cấp sẽ chặn vì không đủ 1.000 user được chấm, và mỗi lần chạy vẫn tốn tiền.

Phương án EC2 với systemd timer nằm trong `deploy/` và được mô tả ở `aws_deployment.md` mục 6. Chỉ chọn nó nếu cần chạy thứ khác trên cùng máy — nếu chỉ để retrain thì EC2 đắt hơn Processing Job.

---

## 9. Giám sát chi phí

### Kiểm tra hằng tuần

Không có endpoint nào — đây là kiểm tra quan trọng nhất, kết quả phải rỗng:

```bash
aws sagemaker list-endpoints --region ap-southeast-1
```

Không có EC2 quên tắt:

```bash
aws ec2 describe-instances --region ap-southeast-1 --filters Name=instance-state-name,Values=running --query "Reservations[].Instances[].[InstanceId,InstanceType]" --output table
```

Bucket không phình:

```bash
aws s3 ls s3://movie-recommendation-fcaj/ --recursive --summarize --human-readable
```

Con số `Total Size` nên quanh 1 GB. Vượt 3 GB nghĩa là lifecycle rule chưa chạy hoặc chưa được tạo.

### Khi hoá đơn tăng bất thường

Mở Cost Explorer, nhóm theo **Service** rồi theo **Usage Type**. Ba nghi phạm theo thứ tự khả năng: SageMaker endpoint ai đó lỡ tạo, EC2 quên tắt, S3 version cũ tích luỹ.

---

## 10. Dọn dẹp

Khi đồ án kết thúc, xoá theo thứ tự này để không sót gì tính tiền.

```bash
aws sagemaker list-endpoints --region ap-southeast-1
```

Có endpoint thì xoá endpoint, endpoint-config và model.

```bash
aws s3 rm s3://movie-recommendation-fcaj --recursive
```

Bucket bật versioning thì lệnh trên **không xoá version cũ**. Phải xoá qua Console (`S3 → bucket → Empty`) hoặc xoá từng version bằng API, rồi mới xoá được bucket:

```bash
aws s3api delete-bucket --bucket movie-recommendation-fcaj --region ap-southeast-1
```

```bash
aws dynamodb delete-table --table-name Movies --region ap-southeast-1
```

Lặp lại cho `PopularMovies`, `Interactions`, `Users`, `RecommendationCache`.

Cuối cùng xoá IAM role và kiểm tra Billing sau 2 ngày để chắc chắn không còn khoản nào phát sinh.

---

## 11. Checklist

Chuẩn bị:

- [ ] Quota `ml.m5.xlarge for processing job usage` ≥ 1
- [ ] Budget alarm đã bật, có email nhận cảnh báo
- [ ] Cost Explorer đã bật
- [ ] `aws sts get-caller-identity` chạy được

Hạ tầng:

- [ ] S3 bucket, chặn public, bật versioning
- [ ] **Lifecycle rule xoá version cũ sau 30 ngày**
- [ ] IAM: SageMaker execution role + inline policy S3
- [ ] IAM: backend role chỉ đọc
- [ ] 5 bảng DynamoDB, tất cả on-demand

Dữ liệu:

- [ ] `aws_sync.py push` xong, S3 khoảng 988 MB
- [ ] `Movies` nạp 45.430 item
- [ ] `PopularMovies` nạp 21 item

Chạy:

- [ ] `--dry-run` cho `source_bundle` = 0.46 MB / 59 file
- [ ] Smoke test `ml.m5.large` chạy xong, đã ghi lại phiên bản Python container
- [ ] Job thật chạy xong, đọc `retrain_report.md`
- [ ] `LATEST.json` trên S3 trỏ đúng version
- [ ] Backend khởi động lại, đọc được artifact mới

Sau khi chạy:

- [ ] `list-endpoints` rỗng
- [ ] Chưa đặt lịch tự động cho tới khi chạy tay thành công một lần
