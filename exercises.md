# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Với temperature từ 0.0 đến 1.0, model vẫn tạo nội dung tương đối mạch lạc và thường chọn sự thật về hang Sơn Đoòng, nhưng cách suy luận và lựa chọn ví dụ có thay đổi. Khi temperature tăng lên 1.5, phản hồi trở nên hỗn loạn, trộn nhiều ngôn ngữ và gần như không còn đáng tin cậy. Điều này cho thấy temperature cao làm tăng tính ngẫu nhiên và đa dạng nhưng cũng tăng nguy cơ sinh nội dung sai hoặc vô nghĩa.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature khoảng 0.2–0.3 cho chatbot hỗ trợ khách hàng. Mức thấp giúp phản hồi nhất quán, bám sát chính sách và hạn chế model tự sáng tạo thông tin không chính xác, trong khi vẫn đủ linh hoạt để diễn đạt tự nhiên.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Workload tạo 10.000 × 3 × 350 = 10.500.000 output token mỗi ngày. Theo bảng giá của bài lab, GPT-4o tốn khoảng 105 USD/ngày, còn GPT-4o-mini tốn khoảng 6,30 USD/ngày, nên GPT-4o đắt hơn khoảng 16,67 lần. GPT-4o xứng đáng với chi phí cho các phân tích phức tạp cần chất lượng cao; mini phù hợp hơn với FAQ, phân loại yêu cầu và các tác vụ đơn giản có lưu lượng lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Trong lần chạy của tôi, phản hồi theo persona giáo viên dài 512 ký tự, dùng hình ảnh chuỗi các viên gạch và các bước đơn giản để giải thích blockchain cho trẻ em. Phản hồi theo persona chuyên gia dài 552 ký tự, sử dụng các thuật ngữ như “decentralized ledger”, “hash”, giao dịch và bên trung gian. Hai phản hồi có độ dài gần nhau do cùng giới hạn output, nhưng khác rõ về từ vựng, mức độ trừu tượng và loại ví dụ. System prompt đã định hướng đối tượng người đọc, giọng điệu và độ chuyên sâu của câu trả lời.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Tôi dùng một đoạn văn tiếng Việt gồm 111 từ: `count_tokens` với encoding của GPT-4o đếm được 131 token, còn công thức số từ / 0,75 ước tính 148 token. Hai kết quả lệch 17 token, tương đương khoảng 11,49% so với số ước tính. Tiếng Việt có dấu và mỗi từ thường gồm nhiều âm tiết tách bằng khoảng trắng; tùy vốn từ của tokenizer, một từ hoặc âm tiết có thể bị chia thành nhiều token con, nên quy đổi cố định từ sang token chỉ là ước lượng.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi model tạo phản hồi dài hoặc có độ trễ cao, vì người dùng thấy nội dung xuất hiện ngay và biết hệ thống vẫn đang hoạt động. Non-streaming phù hợp hơn với phản hồi ngắn, tác vụ chạy nền hoặc trường hợp ứng dụng cần nhận đủ kết quả để kiểm tra, định dạng hay kiểm duyệt trước khi hiển thị.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff tăng dần thời gian chờ sau mỗi lần thất bại, nhờ đó giảm số request gửi tới API và tạo thời gian để dịch vụ phục hồi. Nếu hàng nghìn client cùng retry sau một delay cố định, chúng có thể gửi lại đồng thời và tạo hiệu ứng “thundering herd”, khiến server tiếp tục quá tải. Trong hệ thống thực tế có thể thêm jitter để các client không retry đúng cùng một thời điểm.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona tôi chọn là: “Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt.” Cụm “trợ giảng thân thiện” định hướng model giải thích kiến thức theo cách hỗ trợ và dễ tiếp cận. Yêu cầu “ngắn gọn bằng tiếng Việt” giúp phản hồi tập trung, tiết kiệm token và phù hợp với người học trong khóa.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là trợ lý chỉ giữ ba lượt hội thoại gần nhất nên có thể quên thông tin quan trọng từ đầu phiên. Tôi sẽ cải thiện bằng cách tóm tắt các lượt cũ trước khi cắt history, rồi đưa bản tóm tắt vào context của những lượt tiếp theo. Với ứng dụng lớn hơn, lịch sử và bản tóm tắt có thể được lưu trong cơ sở dữ liệu để truy xuất lại khi cần.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
