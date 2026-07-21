import pytest

from gimmvfi.utils.frame_schedule import fixed_factor_schedule, fps_schedule


def test_fixed_factor_schedule():
    assert fixed_factor_schedule(3, 2) == [
        (0, 0.0),
        (0, 0.5),
        (1, 0.0),
        (1, 0.5),
        (1, 1.0),
    ]


@pytest.mark.parametrize(
    ("input_frames", "source_fps", "target_fps", "expected_frames"),
    [
        (25, 24.0, 60.0, 61),
        (31, 30.0, 60.0, 61),
        (61, 60.0, 24.0, 25),
        (25, 24.0, 24.0, 25),
        (2, 24.0, 60.0, 3),
    ],
)
def test_fps_schedule_frame_count_and_endpoints(
    input_frames, source_fps, target_fps, expected_frames
):
    schedule = fps_schedule(input_frames, source_fps, target_fps)

    assert len(schedule) == expected_frames
    assert schedule[0] == (0, 0.0)
    assert schedule[-1] == (input_frames - 2, 1.0)
    positions = [pair + timestep for pair, timestep in schedule]
    assert positions == sorted(positions)


def test_single_frame_schedule():
    assert fixed_factor_schedule(1, 8) == [(0, 0.0)]
    assert fps_schedule(1, 24.0, 60.0) == [(0, 0.0)]


@pytest.mark.parametrize("source_fps,target_fps", [(0, 60), (24, 0), (-1, 60)])
def test_fps_schedule_rejects_invalid_rates(source_fps, target_fps):
    with pytest.raises(ValueError):
        fps_schedule(2, source_fps, target_fps)
