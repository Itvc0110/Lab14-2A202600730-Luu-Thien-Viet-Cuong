import json
import os

def validate_lab():
    print("🔍 Đang kiểm tra định dạng bài nộp...")

    required_files = [
        "reports/summary.json",
        "reports/benchmark_results.json",
        "analysis/failure_analysis.md"
    ]

    # 1. Kiểm tra sự tồn tại của tất cả file
    missing = []
    for f in required_files:
        if os.path.exists(f):
            print(f"✅ Tìm thấy: {f}")
        else:
            print(f"❌ Thiếu file: {f}")
            missing.append(f)

    if missing:
        print(f"\n❌ Thiếu {len(missing)} file. Hãy bổ sung trước khi nộp bài.")
        return

    # 2. Kiểm tra nội dung summary.json
    try:
        with open("reports/summary.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ File reports/summary.json không phải JSON hợp lệ: {e}")
        return

    if "metrics" not in data or "metadata" not in data:
        print("❌ File summary.json thiếu trường 'metrics' hoặc 'metadata'.")
        return

    metrics = data["metrics"]

    print(f"\n--- Thống kê nhanh ---")
    print(f"Tổng số cases: {data['metadata'].get('total', 'N/A')}")
    print(f"Điểm trung bình: {metrics.get('avg_score', 0):.2f}")

    # EXPERT CHECKS
    has_retrieval = "hit_rate" in metrics
    if has_retrieval:
        print(f"✅ Đã tìm thấy Retrieval Metrics (Hit Rate: {metrics['hit_rate']*100:.1f}%)")
    else:
        print(f"⚠️ CẢNH BÁO: Thiếu Retrieval Metrics (hit_rate).")

    has_multi_judge = "agreement_rate" in metrics
    if has_multi_judge:
        print(f"✅ Đã tìm thấy Multi-Judge Metrics (Agreement Rate: {metrics['agreement_rate']*100:.1f}%)")
    else:
        print(f"⚠️ CẢNH BÁO: Thiếu Multi-Judge Metrics (agreement_rate).")

    if data["metadata"].get("version"):
        print(f"✅ Đã tìm thấy thông tin phiên bản Agent (Regression Mode)")

    metadata = data["metadata"]
    if metadata.get("run_mode") == "real":
        print("Real API mode: enabled")
        if metadata.get("fallback_used"):
            print("❌ Real API report invalid: fallback_used=true")
            return
        api_counts = metadata.get("api_call_counts", {})
        total = metadata.get("total", 0)
        required_api_counts = ["openai_main", "openai_judge", "deepseek_judge"]
        missing_api = [name for name in required_api_counts if api_counts.get(name, 0) < total]
        if missing_api:
            print(f"❌ Real API report invalid: insufficient API calls for {missing_api}")
            return
        if "fallback" in metadata.get("judge_modes", {}):
            print("❌ Real API report invalid: judge_modes contains fallback")
            return
        print("✅ Real API proof fields pass: no fallback and API counts cover all cases")

    print("\n🚀 Bài lab đã sẵn sàng để chấm điểm!")

if __name__ == "__main__":
    validate_lab()
