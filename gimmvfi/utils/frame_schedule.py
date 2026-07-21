def fixed_factor_schedule(input_frames: int, factor: int):
    """Return output entries as (source pair, local timestep)."""
    if input_frames < 1:
        raise ValueError("at least one input frame is required")
    if factor < 1:
        raise ValueError("interpolation factor must be at least 1")
    if input_frames == 1:
        return [(0, 0.0)]

    return [
        (min(index // factor, input_frames - 2), (index % factor) / factor)
        for index in range((input_frames - 1) * factor)
    ] + [(input_frames - 2, 1.0)]


def fps_schedule(input_frames: int, source_fps: float, target_fps: float):
    """Build an endpoint-preserving schedule closest to the requested FPS.

    A finite clip cannot always preserve both endpoints and have an exact target
    frame interval. We preserve the clip duration and choose the nearest output
    frame count, then distribute samples uniformly over that duration.
    """
    if input_frames < 1:
        raise ValueError("at least one input frame is required")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    if input_frames == 1:
        return [(0, 0.0)]

    intervals = max(1, round((input_frames - 1) * target_fps / source_fps))
    source_intervals = input_frames - 1
    schedule = []
    for index in range(intervals + 1):
        source_position = index * source_intervals / intervals
        pair = min(int(source_position), input_frames - 2)
        schedule.append((pair, source_position - pair))
    schedule[-1] = (input_frames - 2, 1.0)
    return schedule
