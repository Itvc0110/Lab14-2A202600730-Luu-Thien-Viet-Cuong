# Reflection - Chi

## Vai trò
Chi phụ trách multi-judge consensus.

## Đóng góp kỹ thuật
- Thiết kế rubric 1-5 cho correctness, grounding, completeness, legal caution và refusal behavior.
- Implement judge panel OpenAI + Mistral với deterministic fallback.
- Tính agreement rate và conflict flag khi hai judge lệch điểm lớn.

## Bài học
Không nên tin một judge duy nhất. Agreement rate giúp biết điểm số có ổn định hay cần review thủ công.
