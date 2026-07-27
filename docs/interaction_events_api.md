# Interaction event contract — cách backend gửi JSON và model chấm điểm

Tài liệu này mô tả hợp đồng giữa backend và module chấm điểm tương tác trong
`src/models/interaction_weights.py` và `src/recommenders/feedback.py`.

Hai điều cần phân biệt ngay:

* **Chấm điểm tương tác** (tài liệu này) chạy trên chuỗi sự kiện thô, không nạp
  artifact, không đọc đĩa. Backend gọi được ở mọi request, kể cả trong Lambda.
* **Sinh gợi ý** (`docs/recommendation_scenarios.md`, `MODEL_DESIGN_SPEC.md`
  mục 13) cần artifact TF-IDF và ALS trong bộ nhớ.

## 1. Bảng event type chính thức

Tám giá trị dưới đây là toàn bộ enum hệ thống chấp nhận. Giá trị ngoài danh sách
**không làm hỏng request**: nó bị bỏ qua, đếm vào `events_ignored` và liệt kê ở
`unsupported_event_types` để backend ghi log.

| `event_type` | `value` gửi kèm | Trọng số | Ghi chú |
|---|---|---:|---|
| `click` | — | +2 | Tò mò, không phải sở thích. Không tính vào lịch sử hợp lệ. |
| `watch` | `float` tỉ lệ đã xem `0.0–1.0` | +10 | Chỉ tính khi `value >= 0.5`. |
| `complete` | — | +12 | Xem hết. |
| `like` | — | +15 | |
| `dislike` | — | −15 | |
| `rating` | `float` số sao `0.5–5.0` | `(value − 3.0) × 5` | 5 sao → +10, 3 sao → 0, 1 sao → −10. |
| **`share`** | — (bỏ qua nếu có) | **+15** | Mới. |
| **`comment`** | `float` sentiment `−1.0–1.0`, hoặc nhãn `positive`/`neutral`/`negative` — **bắt buộc** | **`3 + 12 × sentiment`** | Mới. Không có sentiment thì event bị bỏ, xem mục 1.1. |

Mọi trọng số nằm trong `configs/model_serving.yaml` khối `interactions`, không
hardcode trong code.

### Vì sao `share` bằng đúng `like`

Chia sẻ là đặt tên người dùng lên lời giới thiệu — tín hiệu tự nguyện mạnh nhất
sản phẩm quan sát được. Nhưng payload không cho biết họ chia sẻ để khen hay để
chê, nên nó ngang `like` chứ không cao hơn.

### Vì sao `comment` có công thức riêng

Viết bình luận tốn công thật, nên luôn có phần "engagement" cố định (+3). Phần
còn lại do sentiment quyết định:

```
score = comment_engagement_weight + comment_sentiment_scale × sentiment
      = 3 + 12 × sentiment          (sentiment bị kẹp về [−1, 1])
```

| sentiment | Điểm | Tương đương |
|---:|---:|---|
| `+1.0` | +15 | bằng `like` |
| `+0.5` | +9 | ~ xem gần hết phim |
| `0.0` | +3 | có viết, nhưng không khen |
| `−0.5` | −3 | |
| `−1.0` | −9 | gần bằng `dislike` |
| **không gửi** | **không tính** | event bị bỏ, xem mục 1.1 |

Bình luận gay gắt phải kéo điểm xuống âm, nếu không hệ thống sẽ hiểu "người này
quan tâm phim X" và gợi ý thêm phim giống X — đúng thứ họ vừa chê.

### 1.1. Không có sentiment thì `comment` không được tính — và đó là cố ý

> [!IMPORTANT]
> **Không có giá trị sentiment mặc định.** Bình luận mà backend chưa phân loại
> thái độ thì **bị bỏ hoàn toàn**, đếm vào `events_ignored_by_reason.missing_sentiment`.

Lý do: nếu đặt mặc định dương (bản thiết kế đầu tiên để `+0.25`), thì câu "phim
này dở nhất tôi từng xem" cũng được +6 điểm, và hệ thống sẽ gợi ý thêm phim giống
vậy. Như thế **tệ hơn là không tính gì cả** — nó không chỉ mất tín hiệu mà còn
tạo ra tín hiệu sai dấu.

Hệ quả cần chấp nhận: **chừng nào backend chưa cung cấp sentiment thì `comment`
không đóng góp gì cho model.** Đây là hành vi đúng, không phải thiếu sót cần vá.
Đổi lại, khi backend bổ sung sentiment thì không phải sửa một dòng code nào ở
phía model.

`share` thì khác, và đó là lý do nó vẫn được tính đầy đủ: hành động chia sẻ tự nó
đã là tín hiệu, không cần phân loại thêm.

### 1.2. Lấy sentiment ở đâu — bốn cách, rẻ nhất trước

**a) Số sao gửi kèm bình luận — nên dùng.** Nếu UI cho người dùng vừa chấm sao
vừa viết bình luận (như IMDb, Rotten Tomatoes), thì **số sao chính là sentiment**,
miễn phí và đáng tin hơn mọi mô hình NLP:

```python
def sentiment_from_stars(stars: float) -> float:
    """5 sao -> +1.0, 3 sao -> 0.0, 1 sao -> -1.0."""
    return max(-1.0, min(1.0, (stars - 3.0) / 2.0))
```

Lưu ý: khi đó nên gửi **cả hai** event — `rating` cho số sao và `comment` cho
phần engagement — vì hai event tính điểm riêng và cộng lại.

**b) Thumbs up/down trên chính bình luận.** Nếu UI có nút đánh giá bình luận, dùng
tỉ lệ up/down. Rẻ, không cần NLP, nhưng chỉ có dữ liệu sau khi bình luận đã được
người khác xem.

**c) Quy tắc từ khóa tiếng Việt.** Thô nhưng minh bạch và chạy được ngay: một danh
sách từ tích cực ("hay", "tuyệt", "đáng xem", "xuất sắc") và tiêu cực ("dở", "tệ",
"nhạt", "thất vọng", "lãng phí"), đếm rồi chuẩn hóa về [−1, 1]. Sai nhiều với câu
phủ định ("không hay") và câu châm biếm, nên đặt ngưỡng: chỉ trả sentiment khi số
từ khớp đủ rõ, còn lại trả `null` để event bị bỏ.

**d) Amazon Comprehend.** Xem mục 1.3 — chi phí gần như bằng 0 ở quy mô đồ án,
nhưng phải kiểm tra ngôn ngữ trước.

Code tham chiếu cho cả bốn cách: [`scripts/comment_sentiment_provider.py`](../scripts/comment_sentiment_provider.py).

> [!TIP]
> Với phạm vi đồ án 7 ngày và ngân sách dưới 100 USD, **cách (a) là lựa chọn
> đúng**. Nó không tốn thêm dịch vụ nào, chính xác hơn NLP, và làm xong trong một
> buổi. Nếu UI không có chấm sao kèm bình luận thì tạm thời cứ để `comment` không
> gửi `value` — model bỏ qua và báo lại ở `events_ignored_by_reason`, hệ thống vẫn
> chạy đúng.

### 1.3. Amazon Comprehend: chi phí và cách dùng

#### Cái bẫy phải kiểm tra trước

`DetectSentiment` **chỉ hỗ trợ một tập ngôn ngữ nhất định** — không phải mọi ngôn
ngữ Comprehend nhận diện được. Theo hiểu biết của tôi thì tập đó gồm tiếng Anh,
Đức, Tây Ban Nha, Ý, Bồ Đào Nha, Pháp, Nhật, Hàn, Hindi, Ả Rập, Trung — và
**tiếng Việt không nằm trong đó**. Nhưng AWS bổ sung ngôn ngữ theo thời gian và
tài liệu cũ đi rất nhanh, nên **đừng tin câu trên, hãy hỏi thẳng dịch vụ**:

```bash
python scripts/comment_sentiment_provider.py --check-language --language vi
```

Hoặc bằng AWS CLI:

```bash
aws comprehend detect-sentiment --language-code vi --text "Phim này rất hay" --region ap-southeast-1
```

Hỗ trợ thì trả về `Sentiment` và `SentimentScore`. Không hỗ trợ thì lỗi
`UnsupportedLanguageException` (hoặc boto3 chặn ngay ở client) — và câu trả lời
đó dứt khoát hơn mọi tài liệu.

#### Chi phí

Comprehend tính theo **unit = 100 ký tự, tối thiểu 3 unit mỗi request**. Nghĩa là
bình luận 50 ký tự và bình luận 300 ký tự **giá bằng nhau** — comment ngắn là chỗ
giá thực trên mỗi ký tự tệ nhất.

| Bậc | Giá / unit |
|---|---|
| 10 triệu unit đầu / tháng | 0,0001 USD |
| 10–50 triệu unit | 0,00005 USD |
| trên 50 triệu unit | 0,000025 USD |

**Free tier: 50.000 unit/tháng trong 12 tháng đầu** ≈ **16.600 bình luận/tháng**
miễn phí (bình luận trung bình 200 ký tự = 3 unit).

Kết quả tính thật (`--estimate`):

| Lượng bình luận / tháng | Unit | Giá gốc | Sau free tier |
|---:|---:|---:|---:|
| 1.000 (quy mô demo) | 3.000 | 0,30 USD | **0 USD** |
| 16.600 | ~50.000 | 5,00 USD | **0 USD** |
| 100.000 | 300.000 | 30,00 USD | 25,00 USD |

```bash
python scripts/comment_sentiment_provider.py --estimate 1000
```

**Kết luận về chi phí: ở quy mô đồ án, Comprehend miễn phí.** Chi phí không phải
là lý do để loại nó — ngôn ngữ mới là.

#### Nếu tiếng Việt không được hỗ trợ

Ba lựa chọn, theo thứ tự nên dùng:

1. **Số sao gửi kèm bình luận** (cách a). Miễn phí, chính xác nhất, không thêm
   dịch vụ nào.
2. **Amazon Translate → Comprehend.** Dịch vi→en rồi phân tích. Translate có hỗ
   trợ tiếng Việt. Giá 15 USD/triệu ký tự, free tier 2 triệu ký tự/tháng trong 12
   tháng — quy mô demo vẫn 0 USD, nhưng 100.000 bình luận/tháng thì **300 USD**,
   gấp mười lần Comprehend. Và mất mát thật về độ chính xác: dịch máy hay làm
   phẳng câu châm biếm và câu phủ định kép, đúng chỗ phân tích sentiment vốn đã
   yếu nhất. Phải lấy mẫu kiểm tra trước khi tin.

   ```bash
   python scripts/comment_sentiment_provider.py --text "Phim này dở không tưởng" --translate
   ```

3. **Amazon Bedrock.** Gọi một model ngôn ngữ với prompt phân loại. Hiểu tiếng
   Việt tốt hơn hẳn, nhưng chậm hơn và cần kiểm tra Bedrock có mặt ở
   `ap-southeast-1` cùng quyền truy cập model chưa.

#### Cách dùng

Điểm quan trọng nhất về mặt kỹ thuật: **dùng `Positive − Negative` từ
`SentimentScore`, không dùng nhãn `Sentiment`.** Nhãn làm mất độ lớn — một bình
luận 95% tích cực và một bình luận 55% tích cực đều ra `POSITIVE`, trong khi công
thức `3 + 12 × sentiment` có chỗ dùng cho khác biệt đó.

```python
def score_to_sentiment(scores: dict) -> float:
    """Positive - Negative, kẹp về [-1, 1]. MIXED và NEUTRAL đều ra gần 0."""
    return max(-1.0, min(1.0, scores["Positive"] - scores["Negative"]))


response = comprehend.detect_sentiment(Text=text, LanguageCode="en")
sentiment = score_to_sentiment(response["SentimentScore"])
```

Ba nguyên tắc bắt buộc khi tích hợp:

* **Phân loại thất bại không được làm hỏng việc ghi bình luận.** Comprehend lỗi
  hay hết quota thì cứ lưu bình luận và **bỏ trường `value`**. Model sẽ bỏ qua
  event và báo ở `missing_sentiment` — đúng kết quả như khi chưa có sentiment.
* **Dùng `batch_detect_sentiment` (tối đa 25 văn bản/lần) khi xử lý hàng loạt.**
  Giá mỗi unit y hệt, nhưng số round trip thì không — quan trọng khi chấm lại
  tồn đọng bình luận trước một lần re-train.
* **Lưu sentiment vào DynamoDB ngay lúc ghi bình luận**, đừng gọi lại Comprehend
  mỗi lần cần gợi ý. Sentiment của một bình luận không thay đổi, nên gọi lại là
  trả tiền hai lần cho cùng một câu trả lời.

IAM policy cần thêm cho backend:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["comprehend:DetectSentiment", "comprehend:BatchDetectSentiment"],
    "Resource": "*"
  }]
}
```

Thêm `translate:TranslateText` nếu dùng cách 2.

### Chặn spam: `max_events_per_movie`

Chia sẻ một phim mười lần không nói nhiều hơn chia sẻ một lần, nhưng cộng dồn
không giới hạn sẽ để một phim chiếm hết profile.

```yaml
max_events_per_movie:
  share: 3
  comment: 5
```

Event vượt ngưỡng bị đếm vào `events_capped` và không cộng điểm. Chỉ hai loại
này bị chặn; `click`, `watch`, `rating`… giữ nguyên hành vi cũ.

**Backend phải gửi event mới nhất trước**, vì ngưỡng giữ lại N sự kiện đầu tiên
trong mảng.

## 2. Suy giảm theo thời gian

Mỗi event nhân với `0.5 ^ (số ngày tuổi / half_life_days)`, mặc định
`half_life_days = 30`. Một chuỗi share từ ba tháng trước còn khoảng 12% giá trị
ban đầu. Đây là lý do profile không cần xóa lịch sử cũ thủ công.

## 3. Request

`POST /model/interactions/score` (tên đường dẫn do backend chọn; model chỉ cung
cấp hàm).

```json
{
  "user_id": 12,
  "onboarding_completed": true,
  "as_of": "2026-07-27T12:00:00Z",
  "exclude_movie_ids": [99],
  "events": [
    { "movie_id": 862, "event_type": "share",   "timestamp": "2026-07-27T11:00:00Z" },
    { "movie_id": 862, "event_type": "comment", "value": "positive",
      "timestamp": "2026-07-27T10:55:00Z" },
    { "movie_id": 862, "event_type": "watch",   "value": 0.91,
      "timestamp": "2026-07-26T20:00:00Z" },
    { "movie_id": 550, "event_type": "comment", "value": -0.9,
      "timestamp": "2026-07-25T09:00:00Z" }
  ]
}
```

| Trường | Kiểu | Bắt buộc | Mô tả |
|---|---|---|---|
| `user_id` | int \| string \| null | không | Chỉ dùng để phản chiếu lại và suy ra `scenario_hint`. |
| `onboarding_completed` | bool | không | Mặc định `false`. |
| `as_of` | ISO-8601 | không | Mốc thời gian tính suy giảm. Bỏ trống thì lấy giờ hiện tại. |
| `exclude_movie_ids` | int[] | không | Chỉ dùng bởi `build_recommend_request`. |
| `events` | object[] | **có** | Tối đa `interactions.max_recent_events` (200); phần dư bị cắt, đếm ở `events_truncated`. |
| `events[].movie_id` | int | **có** | Không ép được sang int thì event bị bỏ qua. |
| `events[].event_type` | string | **có** | Xem bảng mục 1. Không phân biệt hoa thường. |
| `events[].value` | number \| string \| null | tùy loại | Xem bảng mục 1. |
| `events[].timestamp` | ISO-8601 | nên có | Thiếu thì event được tính với hệ số suy giảm 1.0. |

Chấp nhận cả tên `recent_interactions` thay cho `events`, để backend dùng chung
một payload với `/model/recommend`.

**`as_of` để làm gì:** không có nó, chấm lại cùng một chuỗi vào hôm sau ra số
khác vì suy giảm tính từ giờ hệ thống. Job batch và test đều cần kết quả lặp
lại được, nên phải truyền `as_of`.

## 4. Response

```json
{
  "user_id": 12,
  "scored_at": "2026-07-27T12:00:00+00:00",
  "half_life_days": 30.0,
  "events_received": 4,
  "events_truncated": 0,
  "events_counted": 4,
  "events_ignored": 0,
  "events_capped": 0,
  "events_ignored_by_reason": {},
  "unsupported_event_types": [],
  "supported_event_types": ["click", "watch", "complete", "like", "dislike",
                            "rating", "share", "comment"],
  "movie_scores": [
    { "movie_id": 862, "score": 39.817078, "normalised_weight": 1.0,
      "disliked": false, "events_counted": 3,
      "event_types": ["comment", "share", "watch"] },
    { "movie_id": 550, "score": -7.79, "normalised_weight": 0.0,
      "disliked": true, "events_counted": 1, "event_types": ["comment"] }
  ],
  "positive_movie_ids": [862],
  "disliked_movie_ids": [550],
  "valid_interaction_count": 4,
  "suggested_scenario_hint": "returning_user"
}
```

| Trường | Backend dùng để làm gì |
|---|---|
| `movie_scores[].score` | Ghi vào profile DynamoDB. **Không so sánh được giữa hai user.** |
| `movie_scores[].normalised_weight` | Điểm chia cho điểm cao nhất, thang `0–1`; dùng khi cần trộn với nguồn khác. |
| `movie_scores[].disliked` | `true` khi điểm ròng ≤ `dislike_score_threshold` (−5.0). |
| `positive_movie_ids` | Danh sách phim để làm seed "vì bạn đã xem", mạnh trước. |
| `disliked_movie_ids` | Đưa thẳng vào `exclude_movie_ids` của lần gọi `/model/recommend` kế tiếp. |
| `valid_interaction_count` | Chính là `valid_interaction_count_90d` trong request gợi ý. |
| `suggested_scenario_hint` | Bảng quyết định mục 10.2 của spec, đã tính sẵn. |
| `events_ignored`, `unsupported_event_types` | **Phải ghi log.** Khác 0 nghĩa là frontend đang bắn event mà model chưa biết. |
| `events_ignored_by_reason` | **Phải ghi log.** Xem bảng dưới. |
| `events_capped` | Khác 0 nghĩa là có người dùng lặp share/comment quá ngưỡng. |

`events_ignored_by_reason` tách lý do ra vì cách xử lý khác nhau:

| Lý do | Nghĩa là gì | Phải làm gì |
|---|---|---|
| `missing_sentiment` | `comment` không kèm sentiment | Bổ sung sentiment ở backend (mục 1.2). Con số này tăng đều nghĩa là toàn bộ bình luận đang không tới được model. |
| `malformed_value` | `value` không ép được sang số | Bug ở frontend hoặc backend. |
| `below_threshold` | `watch` dưới ngưỡng 50% | Bình thường, không cần làm gì. |
| `unsupported_event_type` | Event type ngoài tám enum | Frontend bắn event model chưa biết. |
| `unusable_movie_id` | `movie_id` thiếu hoặc sai kiểu | Bug. |

`score` **không phải** xác suất, không so sánh được giữa hai response, giống
quy tắc ở `MODEL_DESIGN_SPEC.md` mục 13.3.

## 5. `valid_interaction_count` được tính thế nào

```yaml
scenario:
  min_interactions_for_cf: 5
  interaction_recency_days: 90
  valid_event_types: [watch, complete, like, rating, share, comment]
```

Một event được tính khi: type nằm trong `valid_event_types`, **và** nằm trong
90 ngày gần nhất, **và** model thật sự tính được điểm cho nó — tức là `watch`
phải đạt ngưỡng progress, và `comment` phải có sentiment.

Điều kiện cuối quan trọng hơn vẻ ngoài của nó: nếu đếm cả những event mà model
sau đó bỏ qua, người dùng sẽ được nâng lên `returning_user` dựa trên lịch sử
không đóng góp gì, rồi model collaborative không có gì để xếp hạng cho họ.

`click` bị loại vì tò mò không phải lịch sử. `dislike` bị loại vì nó nói cái gì
cần bỏ đi, không nói cái gì để mô hình hóa. `share` và `comment có sentiment`
**được tính**, kể cả comment tiêu cực: chỉ số này đo tài khoản có đủ lịch sử để
chạy collaborative filtering hay không, không đo mức độ hài lòng.

Sự kiện thiếu `timestamp` vẫn được tính. Backend luôn đóng dấu thời gian khi
ghi, nên thiếu dấu nghĩa là dữ liệu import hoặc có lỗi — loại bỏ chúng sẽ âm
thầm hạ cấp người dùng thật xuống onboarding.

## 6. Gọi từ backend

```python
from src.data.config import load_config
from src.recommenders.feedback import (
    InteractionPayloadError,
    build_recommend_request,
    score_interaction_events,
)

# Nạp một lần lúc khởi động tiến trình, không nạp lại mỗi request.
MODEL_CONFIG = load_config("configs/model_serving.yaml")


def score(body: dict) -> dict:
    try:
        return score_interaction_events(body, MODEL_CONFIG)
    except InteractionPayloadError as error:
        raise HTTPException(status_code=400, detail=str(error))
```

Đường tắt cho luồng phổ biến nhất — đọc event của user từ DynamoDB rồi gọi
thẳng model gợi ý:

```python
request = build_recommend_request(body, MODEL_CONFIG, limit=20)
response = engine.recommend(request)     # src/recommenders/engine.py
```

`build_recommend_request` tự điền `scenario_hint`, `valid_interaction_count_90d`
và đẩy `disliked_movie_ids` vào `exclude_movie_ids`, nên phim bị chê vẫn bị lọc
kể cả khi chuỗi event đã bị cắt ngắn qua ngưỡng còn nhìn thấy sự kiện đó.

## 7. Kiểm tra không cần AWS

```bash
python scripts/score_interactions.py --demo
```

```bash
echo '{"user_id":1,"events":[{"movie_id":862,"event_type":"share"}]}' | python scripts/score_interactions.py
```

```bash
python -m pytest tests/test_interaction_events.py -q
```

## 8. Việc còn phải làm ở phía khác

`MODEL_DESIGN_SPEC.md` mục 11.1 yêu cầu: thêm một event type thì phải cập nhật
đồng thời **năm** nơi. Phần model đã xong; bốn phần còn lại chưa:

| Nơi | Trạng thái | Việc cụ thể |
|---|---|---|
| Model + config | **xong** | `SUPPORTED_EVENT_TYPES`, `configs/model_serving.yaml`, tests |
| `MODEL_DESIGN_SPEC.md` mục 11.1 | **chưa** | Mục này đang ghi "không thêm `share`". Phải sửa lại bảng cho khớp tám event type. |
| DynamoDB schema | **chưa** | Bảng `Interactions` cần lưu thêm `value` (sentiment của comment) và `event_type` mới. |
| Backend API | **chưa** | `UserActivityRepository.save_event()` nhận `event_type: str` tự do, chưa validate. Phải chặn giá trị ngoài tám enum và chấp nhận trường `value`. |
| Frontend tracking | **chưa** | Chưa có mã tracking nào; phải bắn `share` khi bấm nút chia sẻ và `comment` khi gửi bình luận. |
| Nguồn sentiment cho `comment` | **chưa** | Chọn một trong bốn cách ở mục 1.2. Chưa có thì `comment` không đóng góp gì — hệ thống vẫn chạy đúng, chỉ là mất tín hiệu đó. |

Model **không** nhận nội dung văn bản bình luận và không phân tích văn bản. Nó chỉ
nhận một con số sentiment. Việc phân loại thái độ nằm hoàn toàn ở phía backend,
nên có thể đổi cách làm (số sao → từ khóa → NLP) mà không phải sửa model.
