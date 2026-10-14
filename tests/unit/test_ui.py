from racing.core.models import CarControl
from racing.render.app import FixedScheduler
from racing.render.input import held_control


def test_controls():
    bindings = tuple(
        (action, (key,))
        for action, key in (
            ("left", "a"),
            ("right", "d"),
            ("accelerate", "w"),
            ("brake", "s"),
            ("brake_override", "space"),
        )
    )
    assert held_control({"a", "d", "w", "s"}, bindings) == CarControl(0.0, 0.0)
    assert held_control({"d", "w", "space"}, bindings) == CarControl(1.0, -1.0)


def test_scheduler_fps_equivalence():
    counts = []
    for fps in (30, 60, 144):
        scheduler = FixedScheduler(0.05)
        ticks = []
        for _ in range(fps * 10):
            scheduler.consume(1 / fps, lambda: ticks.append(len(ticks)))
        assert scheduler.dropped_time_s == 0
        assert scheduler.accumulator < 1e-10
        counts.append(ticks)
    assert counts[0] == counts[1] == counts[2] == list(range(200))


def test_backlog_fraction_and_terminal_clear():
    scheduler = FixedScheduler(0.05)
    ticks = []
    assert scheduler.consume(0.523, lambda: ticks.append(1)) == 5
    assert len(ticks) == 5
    assert abs(scheduler.dropped_time_s - 0.25) < 1e-10
    assert abs(scheduler.accumulator - 0.023) < 1e-10
    assert scheduler.consume(0.1, lambda: False) == 1
    assert scheduler.accumulator == 0
