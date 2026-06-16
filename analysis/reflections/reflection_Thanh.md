# Reflection - Thành

## Vai trò
Thành phụ trách data ingestion và golden dataset.

## Đóng góp kỹ thuật
- Đưa 18 file luật từ Day 7 vào `data/law_corpus/`.
- Tạo `data/corpus.jsonl` bằng recursive chunking.
- Sinh 80 golden cases, gồm seed cases, fact lookup, multi-hop, adversarial, out-of-context và ambiguous cases.

## Bài học
Golden dataset tốt phải có ground-truth retrieval ids. Với RAG pháp luật, chất lượng dataset quyết định khả năng phân biệt lỗi retrieval và lỗi generation.
