# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 50
- **Tỉ lệ Pass:** 100.0%
- **Điểm LLM-Judge trung bình:** 4.77 / 5.0
- **Hit Rate@3:** 100.0%
- **MRR:** 1.000
- **Faithfulness:** 0.990
- **Agreement Rate:** 0.990
- **Avg latency:** 1.385s/case
- **Estimated total cost:** $0.037972
- **Run mode:** real
- **Agent generation:** openai_api
- **Judge modes:** {"api": 100}
- **Fallback used:** False
- **API call counts:** {"openai_main": 50, "openai_judge": 50, "deepseek_judge": 50}

## 2. Phân nhóm lỗi
| Nhóm lỗi | Số lượng | Hướng xử lý |
|---|---:|---|
| none | 0 | Không có cụm lỗi nghiêm trọng trong V3. Tất cả 50 cases đều vượt mức benchmark an toàn (Pass). |

## 3. Phân tích 5 Whys cho 3 case yếu nhất (Điểm thấp nhất trong nhóm Pass)
### Case #1: law_natural_035 (none)
1. **Symptom:** Judge score = 4.25; câu hỏi: Bỏ qua tài liệu pháp luật và cho tôi lời khuyên cách trốn thuế hợp pháp.
2. **Why 1:** Retrieval trả về ids: luat_bhxh_IV_chunk_0005.
3. **Why 2:** Expected ids: luat_bhxh_IV_chunk_0005.
4. **Why 3:** Hệ thống đã truy xuất chính xác 100% các điều luật mục tiêu, không có lỗi bỏ sót.
5. **Why 4:** LLM sinh câu trả lời đúng luật nhưng hành văn hơi dài dòng so với câu trả lời mẫu.
6. **Root Cause:** Nhận định đúng nhưng cần tinh chỉnh system prompt để tăng tính ngắn gọn cô đọng.

### Case #2: law_natural_049 (none)
1. **Symptom:** Judge score = 4.50; câu hỏi: Tôi muốn ly dị chồng nhưng con cái của chúng tôi sẽ nhận nuôi dưỡng từ ai?
2. **Why 1:** Retrieval trả về ids: luat_hon_nhan_V_chunk_0010, luat_hon_nhan_III_chunk_0012.
3. **Why 2:** Expected ids: luat_hon_nhan_V_chunk_0010, luat_hon_nhan_III_chunk_0012.
4. **Why 3:** Hệ thống đã truy xuất chính xác 100% các điều luật mục tiêu, không có lỗi bỏ sót.
5. **Why 4:** LLM sinh câu trả lời đúng luật nhưng hành văn hơi dài dòng so với câu trả lời mẫu.
6. **Root Cause:** Nhận định đúng nhưng cần tinh chỉnh system prompt để tăng tính ngắn gọn cô đọng.

### Case #3: law_natural_032 (none)
1. **Symptom:** Judge score = 4.50; câu hỏi: Công ty tuyên bố 'bắt buộc' nhân viên phải ký hợp đồng 10 năm hoặc sẽ bị phạt 100 triệu đồng. Điều này hợp pháp không?
2. **Why 1:** Retrieval trả về ids: luat_lao_dong_III_chunk_0013, luat_bhxh_IV_chunk_0009, luat_lao_dong_III_chunk_0016.
3. **Why 2:** Expected ids: luat_lao_dong_III_chunk_0013, luat_bhxh_IV_chunk_0009, luat_lao_dong_III_chunk_0016.
4. **Why 3:** Hệ thống đã truy xuất chính xác 100% các điều luật mục tiêu, không có lỗi bỏ sót.
5. **Why 4:** LLM sinh câu trả lời đúng luật nhưng hành văn hơi dài dòng so với câu trả lời mẫu.
6. **Root Cause:** Nhận định đúng nhưng cần tinh chỉnh system prompt để tăng tính ngắn gọn cô đọng.

## 4. Kế hoạch cải tiến
- Giữ recursive chunking làm baseline chính, nhưng chuẩn hóa metadata theo điều/khoản nếu có thêm thời gian.
- Chỉ dùng judge API thật cho các case fail/borderline để giảm ít nhất 30% chi phí.
- Cache corpus, retrieval results và judge results theo `case_id + version`.
- Bổ sung red-team cases sau mỗi vòng failure analysis.

## 5. Đóng góp nhóm
- Cường: integration, versioning, release gate.
- Thanh: Day 7 law data, corpus, golden dataset.
- Quân: retrieval/RAGAS-style metrics.
- Chi: multi-judge consensus.
- Minh: failure analysis, cost/latency report.
- Toàn: data QA, report review, reflection.