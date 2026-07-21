# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# References:
# softmax-splatting: https://github.com/sniklaus/softmax-splatting

import torch
import triton
import triton.language as tl


@triton.jit
def _softsplat_kernel(
    ten_in,
    ten_flow,
    ten_out,
    spatial_elements: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    CHANNEL_BLOCK: tl.constexpr,
    SPATIAL_BLOCK: tl.constexpr,
):
    spatial = tl.program_id(0) * SPATIAL_BLOCK + tl.arange(0, SPATIAL_BLOCK)
    channel = tl.program_id(1) * CHANNEL_BLOCK + tl.arange(0, CHANNEL_BLOCK)
    batch = tl.program_id(2)

    spatial_active = spatial < spatial_elements
    channel_active = channel < channels

    y = spatial // width
    x = spatial % width
    flow_base = batch * 2 * spatial_elements
    flow_x = tl.load(
        ten_flow + flow_base + spatial, mask=spatial_active, other=0.0
    ).to(tl.float32)
    flow_y = tl.load(
        ten_flow + flow_base + spatial_elements + spatial,
        mask=spatial_active,
        other=0.0,
    ).to(tl.float32)

    dst_x = x.to(tl.float32) + flow_x
    dst_y = y.to(tl.float32) + flow_y
    finite = (
        (dst_x == dst_x)
        & (dst_y == dst_y)
        & (tl.abs(dst_x) != float("inf"))
        & (tl.abs(dst_y) != float("inf"))
    )
    dst_x = tl.where(finite, dst_x, 0.0)
    dst_y = tl.where(finite, dst_y, 0.0)

    x0 = tl.floor(dst_x).to(tl.int32)
    y0 = tl.floor(dst_y).to(tl.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    weight_00 = (x1.to(tl.float32) - dst_x) * (y1.to(tl.float32) - dst_y)
    weight_10 = (dst_x - x0.to(tl.float32)) * (y1.to(tl.float32) - dst_y)
    weight_01 = (x1.to(tl.float32) - dst_x) * (dst_y - y0.to(tl.float32))
    weight_11 = (dst_x - x0.to(tl.float32)) * (dst_y - y0.to(tl.float32))

    offsets = (
        (batch * channels + channel[:, None]) * spatial_elements
        + spatial[None, :]
    )
    active = channel_active[:, None] & spatial_active[None, :]
    value = tl.load(ten_in + offsets, mask=active, other=0.0).to(tl.float32)
    output_base = (batch * channels + channel[:, None]) * spatial_elements
    valid = active & finite[None, :]

    tl.atomic_add(
        ten_out + output_base + y0[None, :] * width + x0[None, :],
        value * weight_00[None, :],
        mask=valid
        & (x0[None, :] >= 0)
        & (x0[None, :] < width)
        & (y0[None, :] >= 0)
        & (y0[None, :] < height),
        sem="relaxed",
    )
    tl.atomic_add(
        ten_out + output_base + y0[None, :] * width + x1[None, :],
        value * weight_10[None, :],
        mask=valid
        & (x1[None, :] >= 0)
        & (x1[None, :] < width)
        & (y0[None, :] >= 0)
        & (y0[None, :] < height),
        sem="relaxed",
    )
    tl.atomic_add(
        ten_out + output_base + y1[None, :] * width + x0[None, :],
        value * weight_01[None, :],
        mask=valid
        & (x0[None, :] >= 0)
        & (x0[None, :] < width)
        & (y1[None, :] >= 0)
        & (y1[None, :] < height),
        sem="relaxed",
    )
    tl.atomic_add(
        ten_out + output_base + y1[None, :] * width + x1[None, :],
        value * weight_11[None, :],
        mask=valid
        & (x1[None, :] >= 0)
        & (x1[None, :] < width)
        & (y1[None, :] >= 0)
        & (y1[None, :] < height),
        sem="relaxed",
    )


def _softsplat_forward(
    ten_in: torch.Tensor,
    ten_flow: torch.Tensor,
) -> torch.Tensor:
    if not ten_in.is_cuda or not ten_flow.is_cuda:
        raise ValueError("Triton softsplat requires CUDA tensors")

    if ten_in.ndim != 4:
        raise ValueError(f"ten_in must have shape [N, C, H, W], got {ten_in.shape}")

    if ten_flow.ndim != 4 or ten_flow.shape[1] != 2:
        raise ValueError(f"ten_flow must have shape [N, 2, H, W], got {ten_flow.shape}")

    batch, channels, height, width = ten_in.shape

    if ten_flow.shape != (batch, 2, height, width):
        raise ValueError(
            "ten_in and ten_flow must have matching batch and spatial dimensions"
        )

    ten_in = ten_in.contiguous()
    ten_flow = ten_flow.contiguous()

    # Float32 accumulation avoids half-precision atomics and matches the old
    # custom_fwd behavior under autocast.
    ten_out = torch.zeros(
        (batch, channels, height, width),
        device=ten_in.device,
        dtype=torch.float32,
    )

    spatial_elements = height * width
    channel_block = 4
    spatial_block = 64
    grid = (
        triton.cdiv(spatial_elements, spatial_block),
        triton.cdiv(channels, channel_block),
        batch,
    )
    _softsplat_kernel[grid](
        ten_in,
        ten_flow,
        ten_out,
        spatial_elements=spatial_elements,
        channels=channels,
        height=height,
        width=width,
        CHANNEL_BLOCK=channel_block,
        SPATIAL_BLOCK=spatial_block,
    )

    return ten_out


@torch.compiler.disable()
@torch.no_grad()
def softsplat(
    tenIn: torch.Tensor,
    tenFlow: torch.Tensor,
    tenMetric: torch.Tensor | None,
    strMode: str,
    return_norm: bool = False,
):
    mode_parts = strMode.split("-")
    mode = mode_parts[0]

    if mode not in {"sum", "avg", "linear", "softmax"}:
        raise ValueError(f"unsupported softsplat mode: {strMode}")

    if mode in {"sum", "avg"}:
        if tenMetric is not None:
            raise ValueError(f"tenMetric must be None for mode {mode!r}")
    elif tenMetric is None:
        raise ValueError(f"tenMetric is required for mode {mode!r}")

    if mode == "avg":
        tenIn = torch.cat(
            (
                tenIn,
                torch.ones_like(tenIn[:, :1]),
            ),
            dim=1,
        )
    elif mode == "linear":
        tenIn = torch.cat(
            (
                tenIn * tenMetric,
                tenMetric,
            ),
            dim=1,
        )
    elif mode == "softmax":
        metric = tenMetric.exp()
        tenIn = torch.cat(
            (
                tenIn * metric,
                metric,
            ),
            dim=1,
        )

    tenOut = _softsplat_forward(tenIn, tenFlow)

    if mode == "sum":
        return tenOut

    tenNormalize = tenOut[:, -1:]
    tenOut = tenOut[:, :-1]

    epsilon_mode = mode_parts[1] if len(mode_parts) > 1 else "addeps"

    if epsilon_mode == "addeps":
        tenNormalize = tenNormalize + 1e-7
    elif epsilon_mode == "zeroeps":
        tenNormalize = torch.where(
            tenNormalize == 0.0,
            torch.ones_like(tenNormalize),
            tenNormalize,
        )
    elif epsilon_mode == "clipeps":
        tenNormalize = tenNormalize.clamp_min(1e-7)
    else:
        raise ValueError(f"unsupported epsilon mode: {epsilon_mode}")

    if return_norm:
        return tenOut, tenNormalize

    return tenOut / tenNormalize
