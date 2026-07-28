# Tổng quan hệ thống ML

Tài liệu này mô tả toàn bộ `movie-recommendation-system` — thư mục ML độc lập, không bao gồm frontend và backend. Đọc hết là hiểu được dữ liệu đi từ đâu tới đâu, mô hình nào phục vụ tình huống nào, và vì sao từng lựa chọn thiết kế lại như vậy.

Mọi con số trong tài liệu là số đo thật trên lần chạy đầy đủ ngày 2026-07-28, không phải ước lượng.

---

## 1. Hệ thống làm gì

Gợi ý phim theo kiểu Netflix, phục vụ ba nhóm người dùng khác nhau bằng ba cơ chế khác nhau, gộp kết quả lại thành một danh sách duy nhất cho frontend hiển thị.

Nguyên tắc bao trùm: **batch-first**. Mọi thứ nặng được tính sẵn thành file, lúc có request chỉ đọc từ bộ nhớ. Không có model nào chạy inference thời gian thực, không có endpoint nào bật 24/7. Lý do vừa là tốc độ vừa là tiền: một SageMaker endpoint luôn bật là khoản duy nhất có thể phá ngân sách của dự án.

---

## 2. Dữ liệu nguồn

Bộ dữ liệu là **The Movies Dataset** trên Kaggle (`rounakbanik/the-movies-dataset`), đặt trong `data/movies_dataset_raw/`. Điểm dễ nhầm nhất: đây là **hai nguồn ghép lại**, không phải một.

| File | Dung lượng | Số dòng | Nguồn thật | Vai trò |
|---|---:|---:|---|---|
| `ratings.csv` | 709.6 MB | 26,024,289 | MovieLens | Toàn bộ tín hiệu hành vi |
| `movies_metadata.csv` | 34.4 MB | 45,572 | TMDB | title, overview, poster, genres, vote |
| `credits.csv` | 189.9 MB | 45,476 | TMDB | diễn viên, đạo diễn |
| `keywords.csv` | 6.2 MB | 46,419 | TMDB | từ khoá nội dung |
| `links.csv` | 1.0 MB | 45,843 | MovieLens | cầu nối MovieLens ID ↔ TMDB ID |
| `ratings_small.csv` | 2.4 MB | 100,004 | MovieLens | chỉ để chạy thử `src/models/als_model/als.py` |
| `links_small.csv` | 0.2 MB | 9,125 | MovieLens | đi kèm bản nhỏ |

### Quy ước định danh

Đây là chỗ gây hiểu nhầm nhiều nhất trong dự án, cần nhớ chính xác:

- `movie_id` là **TMDB ID** (lấy từ `movies_metadata.id`)
- `user_id` là **MovieLens ID** (lấy từ `ratings.userId`)
- Đường dịch: `ratings.movieId` → `links.movieId` → `links.tmdbId` → `movies_clean.movie_id`

Hệ quả: output của model mang số hiệu TMDB nên **trông như đến từ TMDB**, nhưng điểm số dùng để xếp hạng thì hoàn toàn từ rating MovieLens. Cách nói chuẩn với người khác: *"xếp hạng bằng rating MovieLens, đánh số phim bằng TMDB ID"*.

TMDB tham gia đúng ba việc, không việc nào là chấm điểm xếp hạng:

1. Cấp `movie_id` làm khoá chính toàn hệ thống
2. Cấp metadata hiển thị (title, overview, poster_path, runtime, genres)
3. Cấp `vote_count` để **loại** phim quá ít vote khỏi bảng phim tương tự

TMDB cũng gián tiếp quyết định **phim nào được vào catalogue**: phim không map được sang TMDB hoặc không có metadata thì bị loại, kéo theo 42,847 dòng rating bị bỏ.

---

## 3. Data pipeline

Chạy bằng một lệnh, sáu giai đoạn theo đúng thứ tự phụ thuộc:

```bash
python scripts/run_data_pipeline.py
```

Orchestration nằm ở [`src/pipeline.py`](../src/pipeline.py).

### Phase B — Profiling dữ liệu thô

`src/data/profiling.py` đọc từng CSV, dò schema, đếm null, phát hiện kiểu dữ liệu sai. Ghi ra `reports/profiling/` và `reports/validation/raw_validation.*`. Mục đích là biết dữ liệu bẩn ở đâu **trước** khi làm sạch, để bước sau không âm thầm nuốt lỗi.

### Phase C — Làm sạch và chuẩn hoá định danh

`src/data/cleaning.py`, phần nặng nhất:

- Chuẩn hoá text (trim, gộp khoảng trắng), ép kiểu, chặn giá trị ngoài miền (`vote_average` phải trong [0,10], `vote_count` không âm)
- Khử trùng lặp phim, ưu tiên bản đầy đủ metadata hơn và nhiều vote hơn
- Dịch ID qua `links.csv`, dựng `id_mapping_clean.parquet`
- Tách bảng chuẩn hoá: genres, keywords, credits, companies, countries

Kết quả:

```
ratings_clean : 25,981,442 dòng   (mất 42,847 ≈ 0.16%)
movies_clean  :     45,430 phim
users_clean   :    270,883 user
```

Bảng bị loại được giữ lại ở `data/interim/rejected_*.parquet` để truy vết chứ không vứt đi.

Phase này cũng **tự sinh** `docs/data_dictionary.md` và `docs/id_mapping.md`. Sửa tay hai file đó là vô nghĩa — lần chạy sau sẽ ghi đè; phải sửa ở `src/data/cleaning.py`.

### Phase D — Feature và ba kịch bản gợi ý

Ba việc song song:

**`src/features/content.py`** — dựng ma trận TF-IDF cho content-based. Ghép overview + keywords + 5 diễn viên chính + đạo diễn + genres + công ty + quốc gia thành một chuỗi text, rồi vector hoá: 30,000 chiều, n-gram 1–2, `min_df=2`, `max_df=0.98`, `sublinear_tf`. Xuất ra `artifacts/content_based/` (`vectorizer.joblib`, `movie_matrix.npz`, `movie_index.parquet`).

**`src/features/interactions.py`** — gom rating theo phim, tính `rating_count` / `average_rating`, xuất `user_item_interactions.parquet` (233 MB) và `movie_rating_stats.parquet`.

**`src/recommenders/guest.py`** — dựng bảng xếp hạng top-rated (mục 4.1).

### Phase E — Chia tập và xuất bản phục vụ

**Chia tập** (`src/data/splitting.py`): theo **thời gian, từng user một**, không ngẫu nhiên. User nào có ≥3 tương tác thì phim cuối cùng làm test, phim áp chót làm validation.

```
interactions_train      : 25,457,092
interactions_validation :    262,175
interactions_test       :    262,175
```

Chia ngẫu nhiên sẽ để model học từ phim user xem tháng 12 rồi đi dự đoán phim tháng 6 — chỉ số đẹp giả tạo và vô dụng.

**Xuất bản phục vụ** (`src/data/serving_export.py`): đóng gói dữ liệu cho backend, gồm `movies_serving.parquet/.jsonl` (metadata hiển thị) và `popular_movies.jsonl` (bảng xếp hạng guest, mỗi dòng là một `(ranking_type, genre)` — khớp đúng khoá chính của bảng DynamoDB `PopularMovies`).

Trường `generated_at` trong `popular_movies.jsonl` là **timestamp rating mới nhất trong dataset** (2017-08-04), không phải ngày build — cố ý, để hai lần chạy trên cùng dữ liệu cho ra file giống hệt nhau.

### Phase F — Kiểm định cuối

`src/data/final_validation.py` chạy một bộ rule và ghi `reports/validation/final_validation.{json,md}`. Rule `critical=true` mà fail là chặn; `critical=false` chỉ cảnh báo.

Trạng thái hiện tại: **WARNING**, do đúng một rule không critical là `KNOWN_DATASET_LIMITATIONS` — nó ghi nhận 2,442 phim thiếu genre, 8,708 user quá thưa để có holdout, và dataset chỉ có `rating`. Đây là **mô tả đặc tính dữ liệu, không phải lỗi**, và không sửa được.

### Tính tái lập

```bash
python scripts/check_determinism.py
```

Chạy lại toàn bộ pipeline rồi so SHA-256 từng artifact. Kết quả: **PASS, 32 artifact giống nhau từng byte**. Điều này quan trọng với job retrain tự động — không tái lập được thì không thể phân biệt "model kém đi vì dữ liệu mới" với "model kém đi vì pipeline chạy khác lần trước".

---

## 4. Ba kịch bản và ba mô hình

Bộ định tuyến ở [`src/recommenders/engine.py`](../src/recommenders/engine.py) quyết định request nào dùng mô hình nào.

### 4.1. Guest — chưa đăng nhập

Không theo dõi, không hồ sơ, không session. Backend gửi `user_id: null` (**không phải chuỗi `"guest"`** — engine ép kiểu int và sẽ trả 4xx).

Model trả về bảng top-rated tính sẵn. Công thức là IMDb weighted rating:

```
score = (v / (v + m)) · R  +  (m / (v + m)) · C
```

- `R` — điểm trung bình của phim
- `v` — số lượt rating của phim
- `C` — điểm trung bình toàn bộ dataset
- `m` — ngưỡng vote tối thiểu, lấy ở **phân vị 90%**

Ý nghĩa: điểm bị kéo về mức trung bình chung, kéo mạnh hay nhẹ tuỳ số lượt vote. Phim 5 sao với 3 lượt bị dìm xuống, phim 4.4 sao với 91 nghìn lượt gần như giữ nguyên.

Tham số đo được trên dữ liệu hiện tại:

```
C = 3.527766          (trung bình 26 triệu rating)
m = 682               (phân vị 90% của số lượt)
Phim đủ điều kiện: 4,546 / 45,430
```

Kết quả top 5:

| # | movie_id | Phim | score | average_rating | rating_count |
|---:|---:|---|---:|---:|---:|
| 1 | 278 | The Shawshank Redemption | 4.4221 | 4.4290 | 91,082 |
| 2 | 238 | The Godfather | 4.3299 | 4.3398 | 57,070 |
| 3 | 629 | The Usual Suspects | 4.2911 | 4.3002 | 59,271 |
| 4 | 424 | Schindler's List | 4.2589 | 4.2665 | 67,662 |
| 5 | 240 | The Godfather: Part II | 4.2496 | 4.2635 | 36,679 |

Cột `score` luôn thấp hơn `average_rating` một chút — đó chính là lực kéo về `C`.

Ngoài bảng `ALL` (100 phim) còn có bảng theo từng thể loại (50 phim/genre), mỗi genre tính `m` riêng. Genre nào có dưới 20 phim đủ điều kiện thì bị bỏ hẳn thay vì xếp hạng bừa.

**Đây không phải machine learning.** Không tham số nào được học, không vòng lặp tối ưu. Chỉ là `GROUP BY` cộng một công thức — viết bằng SQL một câu cũng ra. Dùng chữ "train" cho phần này là sai.

### 4.2. Onboarding — vừa đăng ký, ít lịch sử

Đầu vào là `selected_movie_ids` và `selected_genres` người dùng chọn lúc onboarding. `src/recommenders/content_based.py` dựng vector hồ sơ từ TF-IDF của các phim đã chọn (trọng số 0.70) cộng token thể loại (0.30), rồi tìm phim có cosine cao nhất.

Phim đã chọn bị loại khỏi kết quả. Input rỗng hoặc không dùng được thì rơi về bảng guest.

Nhánh này giải quyết cold-start: không cần biết ai đã rating gì, chỉ cần metadata.

### 4.3. Returning user — có lịch sử thật

Đây là nhánh **hybrid**, chạy ba nguồn song song rồi hợp nhất. Chi tiết ở mục 6.

---

## 5. Mô hình ALS — thứ duy nhất thực sự được huấn luyện

`src/models/collaborative.py` + `train.py`.

### Vì sao là implicit chứ không phải explicit

Dataset chỉ có rating, nhưng hệ thống thật sẽ thu click/watch/like. Nếu train dạng explicit (dự đoán số sao) thì lúc lên production không có sao mà dự đoán. Nên rating được **dịch sang tín hiệu implicit** ngay từ đầu, để mô hình khớp với dữ liệu tương lai.

### Luật chuyển đổi

| Rating | Xử lý | Số dòng | Tỷ lệ |
|---|---|---:|---:|
| ≥ 4.0 | positive, độ tin theo sao | **12,659,286** | 49.7% |
| 2.5 < r < 4.0 | neutral → **vứt bỏ** | 8,216,557 | 32.3% |
| ≤ 2.5 | negative → không vào ma trận | 4,581,249 | 18.0% |
| không có | *chưa biết*, tuyệt đối không coi là ghét | — | — |

Độ tin cậy `c = 1 + α·w` với `α = 40`:

```
4.0 sao → 11      4.5 sao → 21      5.0 sao → 31
```

Rating âm **không** được biểu diễn bằng confidence âm — thư viện `implicit` không có khái niệm đó. Chúng thành **danh sách loại trừ riêng từng user**, áp dụng lúc chọn ứng viên.

Phim có dưới 5 lượt positive bị loại: 13,909 phim, còn 17,608.

### Ma trận và huấn luyện

```
Ma trận : 262,571 user × 17,608 phim
Ô khác 0: 12,634,032
Sparsity: 99.726734%
```

262,571 ít hơn 270,883 user ban đầu — khoảng 8,300 user không có nổi một rating ≥4.0 nên biến mất. Đó là lúc `knows_user()` trả `False` và engine tự hạ scenario xuống onboarding.

```python
AlternatingLeastSquares(factors=64, regularization=0.05,
                        iterations=20, random_state=42)
```

ALS xấp xỉ ma trận thành tích hai ma trận nhỏ `R ≈ U(262,571×64) × Vᵀ(64×17,608)`. 64 chiều ẩn do model tự tìm, không ai đặt tên. "Alternating" là cách giải: cố định `V` giải `U`, cố định `U` giải `V`, lặp 20 vòng — mỗi vòng là bài toán bình phương tối thiểu có nghiệm đóng, giải thẳng không cần gradient descent.

**Thời gian train thực đo: 31.2 giây.**

`train.py` khoá `OPENBLAS_NUM_THREADS=1` **trước khi** import implicit, vì thread pool của implicit và của BLAS sẽ tranh CPU của nhau.

### Artifact

```
artifacts/collaborative/v1.0.0/
├── user_factors.npy      (262,571 × 64)
├── item_factors.npy      ( 17,608 × 64)
├── user_index.parquet    row_index → user_id
├── movie_index.parquet   row_index → movie_id
├── config.json
└── manifest.json
```

Hai file index sống còn: factor chỉ là mảng số, `row 5` là user nào phải tra index mới biết. Lệch index sẽ trả ra ID hợp lệ nhưng là **phim khác** — sai mà không báo lỗi.

`manifest.json` ghi git commit, hyperparameter, thống kê dữ liệu, thời gian train, phiên bản thư viện, và metrics do `evaluate.py` ghi ngược vào sau khi đo.

### Lúc phục vụ

Không nạp dữ liệu train, chỉ nạp hai mảng factor:

```python
scores = user_factors[row] @ item_factors.T   # → 17,608 điểm
```

Một phép nhân ma trận ra điểm toàn bộ catalogue, rồi `argpartition` lấy top-N. Phim bị loại trừ được gán `-inf` trước khi lấy top. Chấm theo lô 512 user để không cấp phát vài GB một lúc.

---

## 6. Tương tác thời gian thực

`src/models/interaction_weights.py` gom một luồng sự kiện hỗn hợp thành **một điểm số cho mỗi phim**, để tầng xếp hạng không cần biết gì về loại sự kiện.

### Tám loại event và trọng số

| Event | Điểm | Ghi chú |
|---|---:|---|
| `click` | 2 | mở trang chi tiết chỉ là tò mò |
| `watch` | 10 | chỉ tính khi xem quá 50% thời lượng |
| `complete` | 12 | xem hết |
| `like` | **15** | |
| `share` | 15 | ngang like — có người share để chê |
| `dislike` | −15 | |
| `rating` | `(sao − 3) × 5` | 5 sao = +10, 3 sao = 0, 1 sao = −10 |
| `comment` | `3 + 12 × sentiment` | cần backend gắn sentiment |

Đáng chú ý: **một `like` (15) nặng hơn một rating 5 sao (10)**. Người chỉ bấm like mà không bao giờ chấm sao không hề bị thiệt. Ngược lại rating là thứ duy nhất mang **độ lớn** — 1 sao khác hẳn 3 sao — nên hai cơ chế bổ sung cho nhau, không thay thế nhau.

`comment` **không có sentiment mặc định**. Comment chưa được phân loại bị bỏ hẳn và đếm vào `ignored_by_reason.missing_sentiment`. Coi nó là hơi tích cực sẽ khiến "phim dở nhất tôi từng xem" bị đọc thành hứng thú và gợi ý thêm phim tương tự — tệ hơn là bỏ qua.

### Ba quy tắc thiết kế

**Suy giảm theo thời gian.** Mỗi sự kiện nhân `0.5 ^ (số_ngày / 30)`. Sở thích 30 ngày trước chỉ còn tính một nửa.

**Không tương tác ≠ ghét.** Phim chưa đụng tới là *chưa biết*. Chỉ tín hiệu âm rõ ràng mới ra điểm âm.

**Nỗ lực khác tán thành.** `share` và `comment` tốn công hơn click nhiều nên nặng ký, nhưng comment có thể là chửi — trọng số của nó đi theo sentiment và được phép âm.

Riêng `share` và `comment` bị chặn trần (`max_events_per_movie`: 3 và 5). Share một phim cho mười người không nói gì hơn share một lần, mà tổng không giới hạn sẽ để một phim nuốt hết hồ sơ.

### Ví dụ thật

Input: like phim 550 (0.58 ngày trước), watch 90% phim 550, click phim 278 (7.58 ngày trước), dislike phim 238 (1.58 ngày trước).

```
scores       : {550: 24.66, 278: 1.68, 238: -14.46}
disliked     : {238}
valid_count  : 2  → scenario: onboarding_user
```

Diễn giải `24.66` = `like 15 × 0.9866` + `watch 10 × 0.9857`.

`valid_count` chỉ đếm 2 vì `click` và `dislike` **cố tình không nằm** trong `scenario.valid_event_types` — tò mò không phải lịch sử xem, và dislike nói cho hệ thống biết cái gì cần bỏ chứ không phải cái gì cần học.

### Backend dùng chung luật

`src/recommenders/feedback.py` là hàm thuần, không đọc artifact, không chạm đĩa. Backend gọi nó để:

- chấm điểm luồng event rồi lưu hồ sơ vào DynamoDB
- tính `valid_interaction_count_90d` và chọn `scenario_hint` **trước khi** gọi model

Luật sống ở `configs/model_serving.yaml` và được phục vụ ngược ra, thay vì để backend cài lại — hai định nghĩa lệch nhau sẽ âm thầm định tuyến user sang sai mô hình.

Bảng quyết định:

```
user_id rỗng            → guest
chưa xong onboarding    → guest
valid_count < 5         → onboarding_user
còn lại                 → returning_user
```

---

## 7. Tầng hybrid

`src/models/hybrid_ranking.py`.

### Hợp nhất bằng thứ hạng, không phải điểm

Ba nguồn cho điểm trên ba thang không so được: weighted rating 0–5, cosine 0–1, ALS dot product vô hạn. Cộng thẳng thì nguồn có biên độ lớn nhất nuốt hai nguồn kia. Weighted RRF vứt hết điểm, chỉ giữ vị trí:

```
rrf_score(m) = Σ_s  w_s · 1 / (60 + rank_s(m))
```

Phim vắng mặt ở một nguồn thì đóng góp 0 từ nguồn đó, nên có mặt ở hai danh sách thực sự thắng có mặt ở một.

### Trọng số trượt theo lịch sử

| Số tương tác | w_cf (ALS) | w_cb (TF-IDF) | w_pop |
|---:|---:|---:|---:|
| 5 | 0.25 | **0.81** | 0.00 |
| 10 | 0.50 | **0.63** | 0.00 |
| 15 | 0.75 | 0.44 | 0.00 |
| 20+ | **1.00** | 0.25 | 0.00 |

User vừa đủ 5 tương tác thì **nhánh nội dung nặng gấp hơn 3 lần nhánh ALS**; phải tới 20 tương tác ALS mới thắng. Bàn giao mượt là cố ý — ALS chưa biết gì về user mới, ép tin nó sẽ ra rác.

`w_pop = 0.00`: popularity **không tham gia hợp nhất**, đo trên validation thấy không cải thiện gì. Nó chỉ còn vai trò lớp chữa cháy cuối.

### Sau khi hợp nhất

1. **Nudge theo phim vừa xem**: cộng `0.15 × cosine` giữa ứng viên và hồ sơ TF-IDF có trọng số — phim user xem hết kéo mạnh hơn phim chỉ click.
2. **Trần đa dạng**: tối đa 4 phim cùng thể loại chính trong top 20. Phim vượt trần bị đẩy xuống cuối hàng đợi chứ **không bị loại**; phim thiếu genre không bao giờ bị chặn, vì phạt dữ liệu thiếu sẽ âm thầm giấu mất một phần catalogue.
3. **Backfill**: thiếu thì độn thêm theo thứ tự content → genre → global. Bốn kết quả cá nhân hoá cộng mười sáu phim phổ biến vẫn tốt hơn hai mươi phim phổ biến.

---

## 8. Hợp đồng API

### Request

```json
{
  "user_id": 12,
  "scenario_hint": "returning_user",
  "onboarding_completed": true,
  "valid_interaction_count_90d": 37,
  "selected_movie_ids": [],
  "selected_genres": ["Drama"],
  "recent_interactions": [
    {"movie_id": 550, "event_type": "like", "timestamp": "..."},
    {"movie_id": 862, "event_type": "watch", "value": 0.82, "timestamp": "..."}
  ],
  "exclude_movie_ids": [],
  "limit": 20
}
```

`build_recommend_request()` trong `src/recommenders/feedback.py` dựng sẵn body này từ luồng event lưu trong DynamoDB.

### Response

```json
{
  "model_name": "hybrid_recommender",
  "model_version": "1.0.0",
  "scenario_applied": "guest",
  "recommendation_type": "top_rated",
  "fallback_used": false,
  "fallback_level": "none",
  "generated_at": "2026-07-28T05:30:00+00:00",
  "artifact_versions": {"content_based": "local", "collaborative": "v1.0.0"},
  "recommendations": [
    {"movie_id": 278, "score": 1.0, "reason_code": "top_rated", "reason_context": {}}
  ]
}
```

Ba điều bắt buộc phải hiểu đúng:

**Response không bao giờ chứa title, overview hay poster.** Backend tự join từ catalogue của mình. Metadata đổi liên tục còn ranking tính theo batch — nhét chung thì đổi poster cũng phải chạy lại pipeline.

**`score` không phải điểm phim.** Ba nhánh cho điểm ở ba thang khác nhau, chỉ **thứ tự** có nghĩa; giá trị được chuẩn hoá về `(0,1]` cho nhất quán. Frontend đừng hiển thị nó thành "độ khớp 87%".

**`scenario_hint` vào ≠ `scenario_applied` ra.** Backend gợi ý vì chỉ nó biết trạng thái tài khoản; engine có thể hạ cấp vì chỉ nó biết artifact có phục vụ nổi không. Hai sự thật đi ở hai trường riêng để không bên nào phải đoán ý bên kia.

`reason_code` có các giá trị: `top_rated`, `top_rated_genre`, `similar_to_onboarding`, `genre_match`, `similar_users`, `similar_to_watched_movies`.

### Hàng "Vì bạn đã xem X"

`because_you_watched()` là lookup thuần từ bảng tính sẵn `artifact_content_based/similar_movies_top50.parquet`, không gọi model, không nhân ma trận. Bảng hiện có **2,271,378 dòng**, 13,798 phim đủ điều kiện làm hàng xóm, **0 phim không có hàng xóm**.

Phim ít vote bị loại khỏi tập ứng viên hàng xóm (`min_vote_count: 25`) vì vector TF-IDF ngắn cho cosine cao giả tạo — đó là lý do "The Shawshank Redemption" từng khớp với một phim nhà tù không có vote nào.

---

## 9. Đánh giá

```bash
python evaluate.py --sample-users 5000
```

Đo trên tập test, mỗi user giấu đúng một phim.

| Model | Users | HitRate@10 | HitRate@20 | NDCG@10 | Phim khác nhau | Catalogue |
|---|---:|---:|---:|---:|---:|---:|
| `popularity_train` | 5,000 | 0.0332 | 0.0592 | 0.0201 | 128 | 0.28% |
| **`collaborative_als`** | 4,943 | **0.1115** | **0.1784** | **0.0537** | 2,411 | 5.31% |
| `content_tfidf` | 4,943 | 0.0051 | 0.0069 | 0.0035 | 17,530 | 38.59% |
| `hybrid_rrf` | 5,000 | 0.0818 | 0.1394 | 0.0393 | 8,110 | 17.85% |

- ALS vượt baseline popularity **+235.8%**
- Hybrid vượt **+146.4%**
- Content-based đơn lẻ **thua** baseline (−84.8%) — đúng như kỳ vọng: nó tồn tại để giải cold-start, không phải để thắng trên user có lịch sử dày

Vì mỗi user chỉ giấu một item, `Recall@K` bằng `HitRate@K` và `Precision@K` bằng `HitRate@K / K` — hai cột đó không mang thông tin độc lập.

Đổi lại độ chính xác là độ phủ: ALS chỉ chạm 5.31% catalogue, hybrid 17.85%. Trần đa dạng thể loại tồn tại một phần để cân lại chuyện này.

---

## 10. Phiên bản và retrain

### Con trỏ phiên bản

`artifacts/LATEST.json` trỏ tới artifact đang phục vụ. Engine đọc nó lúc khởi động. Đổi model = đổi một dòng JSON, không deploy lại code.

### Chu trình retrain

```bash
python retrain.py --version v1.1.0 --events s3://bucket/events/2026-07-27/ --pull --push
```

Sáu bước, giống hệt nhau trên laptop, trong SageMaker Processing Job và trên EC2:

1. Kéo dataset và artifact hiện tại từ S3 (`--pull`)
2. Nạp event mới, quy về thang rating 0.5–5.0 rồi ghép vào bảng interaction
3. Dựng lại split theo thời gian trên tập hợp nhất
4. Train artifact ALS mới dưới phiên bản mới
5. Đánh giá so với baseline popularity và model đang chạy
6. Thăng cấp trong `LATEST.json`, hoặc giữ nguyên con trỏ cũ (`--push`)

Khi dựng dữ liệu train, **decay bị tắt** (`half_life_days: 0`). Decay trả lời "hôm nay user muốn gì" — đó là câu hỏi lúc phục vụ. Lúc train cần toàn bộ lịch sử được cân như nhau mỗi lần chạy, nếu không tập train sẽ phụ thuộc đồng hồ và hai lần chạy trên cùng dữ liệu ra hai model khác nhau.

User vừa rating vừa tương tác một phim sẽ sinh hai dòng ứng viên; `event_conflict_policy: prefer_rating` cho rating thắng, vì đó là tín hiệu mô hình được thiết kế quanh nó.

### Cổng thăng cấp

Retrain chạy theo lịch, không ai duyệt từng lần. Không có cổng thì một lần export feedback hỏng sẽ âm thầm thay mất model đang chạy tốt, và dấu hiệu đầu tiên là người dùng phàn nàn.

Ba kiểm tra độc lập, mỗi cái chỉ có quyền **chặn**:

1. Đủ số user được chấm (`minimum_users_scored: 1000`) — dưới ngưỡng thì con số là nhiễu
2. Vượt baseline popularity đo **trong cùng lần chạy** — model cá nhân hoá mà không thắng nổi "ai cũng xem phim này" thì không đáng phục vụ
3. Không tụt quá 5% so với model đang chạy — là dung sai chứ không phải yêu cầu tăng, vì retrain trên dữ liệu mới làm chỉ số dao động vài phần trăm cả hai chiều

Có thêm cảnh báo `protocol_mismatch`: hai chỉ số đo trên số user chênh nhau quá 25% thì không so sánh được, và điều đó **không** mở cổng — nó nói rằng phán quyết không phải bằng chứng theo chiều nào cả.

---

## 11. Triển khai AWS

Kiến trúc: artifact tĩnh trên **S3**, bảng tra cứu trên **DynamoDB**, retrain định kỳ bằng **SageMaker Processing Job**. Không có endpoint thời gian thực.

### Năm bảng DynamoDB

| Bảng | Khoá | Nội dung |
|---|---|---|
| `Movies` | `movie_id` | metadata hiển thị |
| `Interactions` | `user_id` + `sk` | event người dùng; `sk` = `interaction_timestamp#movie_id` |
| `PopularMovies` | `ranking_type` + `genre` | bảng xếp hạng guest |
| `Users` | `user_id` | hồ sơ |
| `RecommendationCache` | `user_id` + `scenario` | cache kết quả |

Không có bảng `SimilarMovies`. Bảng phim tương tự là ma trận tĩnh 45k×50, nằm trên **S3** và được nạp vào RAM lúc process khởi động — nhét vào DynamoDB chỉ tốn tiền ghi mà không lợi gì.

### Vì sao Processing Job

- Lần chạy không chỉ là fit: nó nạp event, dựng lại split, đánh giá và quyết định thăng cấp — đúng hình dạng của một script
- Artifact phục vụ là cặp `.npy` đọc thẳng từ S3, không phải SageMaker Model, nên phần đóng gói của Training Job không mang lại gì
- Endpoint luôn bật là khoản duy nhất có thể vượt ngân sách

Job tự kéo input từ S3 và tự đẩy output lên, thay vì dùng kênh ProcessingInput/ProcessingOutput — dataset nằm trong cây thư mục lồng nhau mà code đã biết, đi vòng qua thư mục kênh sẽ thành hai quy ước đường dẫn cho cùng một tập tin.

### Đóng gói mã nguồn

`scripts/sagemaker_retrain_job.py` dựng thư mục staging chỉ chứa mã (**0.46 MB, 59 file**) rồi mới upload. Trỏ `source_dir` thẳng vào gốc repo sẽ nén cả `data/` 1.7 GB và virtualenv — khoảng 2.2 GB mỗi lần submit, mà job đã tự `--pull` dữ liệu từ S3 rồi.

Container cài `requirements-container.txt` (lower bound) chứ không phải `requirements.txt` (pin chính xác). Bản pin đòi Python ≥3.11; base image ship interpreter nào là tuỳ phiên bản framework, và pin cứng biến chuyện lệch Python thành job chết lúc `pip install`. `deploy/sagemaker_retrain.py` in phiên bản Python nó nhận được ngay dòng đầu log.

---

## 12. Cấu trúc thư mục

```
movie-recommendation-system/
├── configs/
│   ├── data_pipeline.yaml     # đường dẫn, ngưỡng làm sạch, tham số TF-IDF, split
│   ├── model_serving.yaml     # ALS, hybrid, trọng số event, scenario
│   └── aws.yaml               # region, bucket, DynamoDB, SageMaker, retrain
├── data/
│   ├── movies_dataset_raw/    # 7 file CSV gốc từ Kaggle
│   ├── interim/               # dòng bị loại, để truy vết
│   ├── processed/             # bảng đã làm sạch
│   ├── features/              # TF-IDF, ma trận tương tác
│   ├── splits/                # train / validation / test
│   ├── serving/               # dữ liệu cho backend và DynamoDB
│   └── samples/               # ví dụ schema
├── artifacts/
│   ├── collaborative/<ver>/   # factor ALS + index + manifest
│   ├── content_based/         # vectorizer, ma trận, bảng phim tương tự
│   ├── eval/                  # baseline popularity
│   └── LATEST.json            # con trỏ phiên bản đang phục vụ
├── src/
│   ├── data/                  # cleaning, splitting, serving export, feedback ingest
│   ├── features/              # content TF-IDF, interactions, similarity
│   ├── models/                # collaborative (ALS), hybrid_ranking, interaction_weights
│   ├── recommenders/          # engine, guest, content_based, feedback
│   ├── aws/                   # đồng bộ S3
│   └── pipeline.py            # orchestration
├── scripts/                   # từng bước pipeline chạy riêng được
├── deploy/                    # entrypoint SageMaker, systemd unit cho EC2
├── tests/
├── train.py                   # huấn luyện ALS
├── evaluate.py                # đo so với baseline
├── inference.py               # thử nhanh
└── retrain.py                 # chu trình retrain 6 bước
```

Quy tắc chung của `configs/`: **mọi khoá trong file đều được code đọc**. Khoá cho thành phần chưa làm chỉ được thêm khi thành phần đó ra đời, để file không bao giờ quảng cáo một hợp đồng hệ thống không tôn trọng.

---

## 13. Chạy lại từ đầu

```bash
pip install -r requirements.txt
# đặt 7 file CSV vào data/movies_dataset_raw/

python scripts/run_data_pipeline.py      # ~5 phút
python scripts/build_similar_movies.py   # ~78 giây
python train.py --version v1.0.0         # ~31 giây
python evaluate.py --sample-users 5000
```

Kiểm tra:

```bash
pytest                                   # 63 test
python scripts/check_determinism.py      # so hash 32 artifact
python scripts/demo_cli.py               # thử gợi ý bằng tay
```

Phần AWS cần thêm `pip install -r requirements-aws.txt`, một S3 bucket, và một IAM role cho SageMaker.

---

## 14. Giới hạn đã biết

Ghi ra đây để không ai phải phát hiện lại:

- **Dataset chỉ có rating.** Toàn bộ cơ chế 8 loại event là chuẩn bị cho production; offline không có gì để kiểm chứng nó ngoài rating.
- **Dữ liệu dừng ở 2017.** `generated_at` trong bảng xếp hạng phản ánh đúng điều đó.
- **2,442 phim không có genre**, nên không vào được bảng xếp hạng theo thể loại và không bị trần đa dạng chặn.
- **8,708 user quá thưa** để có holdout, chỉ nằm trong train.
- **~8,300 user không có rating ≥4.0 nào** nên vắng mặt khỏi ALS; họ luôn bị hạ xuống onboarding.
- **`comment` chưa đóng góp gì** cho tới khi backend cung cấp sentiment. Đây là hành vi cố ý, không phải lỗ hổng cần vá.
- **`src/models/als_model/als.py`** là bản ALS tự cài để nộp bài học thuật, dùng ma trận dense. Chạy trên dataset thật cần khoảng 97 GB RAM, nên nó chỉ dùng được với `ratings_small.csv`; hệ thống thật dùng thư viện `implicit`.
