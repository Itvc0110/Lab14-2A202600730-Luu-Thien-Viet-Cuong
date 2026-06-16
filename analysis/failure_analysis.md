# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 80
- **Tỉ lệ Pass:** 87.5%
- **Điểm LLM-Judge trung bình:** 4.60 / 5.0
- **Hit Rate@3:** 97.5%
- **MRR:** 0.935
- **Faithfulness:** 1.000
- **Agreement Rate:** 0.983
- **Avg latency:** 0.005s/case
- **Estimated total cost:** $0.001981

## 2. Phân nhóm lỗi
| Nhóm lỗi | Số lượng | Hướng xử lý |
|---|---:|---|
| none | 5 | Ưu tiên kiểm tra retrieval trace và prompt theo nhóm lỗi này. |
| retrieval_miss | 2 | Ưu tiên kiểm tra retrieval trace và prompt theo nhóm lỗi này. |
| wrong_domain | 4 | Ưu tiên kiểm tra retrieval trace và prompt theo nhóm lỗi này. |

## 3. Phân tích 5 Whys cho 3 case yếu nhất
### Case #1: law_seed_003 (none)
1. **Symptom:** Judge score = 2.29; câu hỏi: Mức trợ cấp một lần khi nghỉ hưu được quy định như thế nào đối với người có số năm đóng bảo hiểm xã hội cao hơn mức tối đa hưởng lương hưu?
2. **Why 1:** Retrieval trả về ids: luat_bhxh_V_chunk_0040, luat_bhxh_III_chunk_0003, luat_bhxh_V_chunk_0063.
3. **Why 2:** Expected ids: luat_bhxh_V_chunk_0040.
4. **Why 3:** Nếu chunk đúng không nằm top-k, nguyên nhân nằm ở lexical matching/chunk boundary.
5. **Why 4:** Nếu answer thiếu ý, prompt cần ép trả lời theo điều kiện/ngoại lệ từ context.
6. **Root Cause:** none.

### Case #2: law_seed_005 (none)
1. **Symptom:** Judge score = 2.67; câu hỏi: Mức trợ cấp mai táng đối với thân nhân của người lao động tham gia bảo hiểm xã hội bắt buộc bị chết là bao nhiêu?
2. **Why 1:** Retrieval trả về ids: luat_bhxh_V_chunk_0058, luat_bhxh_V_chunk_0060, luat_bhxh_I_chunk_0007.
3. **Why 2:** Expected ids: luat_bhxh_V_chunk_0058.
4. **Why 3:** Nếu chunk đúng không nằm top-k, nguyên nhân nằm ở lexical matching/chunk boundary.
5. **Why 4:** Nếu answer thiếu ý, prompt cần ép trả lời theo điều kiện/ngoại lệ từ context.
6. **Root Cause:** none.

### Case #3: law_seed_004 (none)
1. **Symptom:** Judge score = 2.75; câu hỏi: Công dân Việt Nam cần đáp ứng điều kiện gì để được hưởng trợ cấp hưu trí xã hội?
2. **Why 1:** Retrieval trả về ids: luat_bhxh_III_chunk_0003, luat_bhxh_III_chunk_0001, luat_bhxh_I_chunk_0014.
3. **Why 2:** Expected ids: luat_bhxh_III_chunk_0001.
4. **Why 3:** Nếu chunk đúng không nằm top-k, nguyên nhân nằm ở lexical matching/chunk boundary.
5. **Why 4:** Nếu answer thiếu ý, prompt cần ép trả lời theo điều kiện/ngoại lệ từ context.
6. **Root Cause:** none.

## 4. Kế hoạch cải tiến
- Giữ recursive chunking làm baseline chính, nhưng chuẩn hóa metadata theo điều/khoản nếu có thêm thời gian.
- Chỉ dùng judge API thật cho các case fail/borderline để giảm ít nhất 30% chi phí.
- Cache corpus, retrieval results và judge results theo `case_id + version`.
- Bổ sung red-team cases sau mỗi vòng failure analysis.

## 5. Đóng góp nhóm
- Cường: integration, versioning, release gate.
- Thành: Day 7 law data, corpus, golden dataset.
- Quân: retrieval/RAGAS-style metrics.
- Chi: multi-judge consensus.
- Minh: failure analysis, cost/latency report.
- Toàn: data QA, report review, reflection.