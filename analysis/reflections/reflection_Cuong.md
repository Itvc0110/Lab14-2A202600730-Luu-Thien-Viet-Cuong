# Reflection - Cường

## Vai trò
Cường phụ trách integration, versioning và release gate cho Lab 14.

## Đóng góp kỹ thuật
- Tích hợp pipeline chạy benchmark cho `V1_Base`, `V2_RetrievalPlus`, `V3_CostAware`.
- Chuẩn hóa output `reports/summary.json` và `reports/benchmark_results.json`.
- Thiết kế release gate dựa trên score, hit rate, MRR và cost growth.

## Bài học
Regression testing giúp chứng minh cải thiện bằng số liệu thay vì cảm tính. V3 tốt hơn cho production vì giữ chất lượng như V2 nhưng giảm chi phí ước tính.
