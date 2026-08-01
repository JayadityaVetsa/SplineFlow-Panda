from splineflow_panda.benchmark import summarize_benchmark


def test_benchmark_summary_reports_speed_success_frontier() -> None:
    rows = [
        {
            "representation": representation,
            "speedup": speed,
            "success": success,
            "completion_time_s": 4 / speed,
        }
        for representation, values in {
            "action_chunk": [(1, True), (2, True), (3, False)],
            "bspline_action": [(1, True), (2, True), (3, True)],
        }.items()
        for speed, success in values
    ]
    report = summarize_benchmark(rows, minimum_success_rate=0.8)
    assert report["action_chunk"]["maximum_reliable_speedup"] == 2
    assert report["bspline_action"]["maximum_reliable_speedup"] == 3
    assert report["bspline_action"]["by_speedup"]["2.0"]["achieved_speedup"] == 2
