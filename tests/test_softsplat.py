import pytest
import torch

from gimmvfi.generalizable_INR.modules.softsplat import softsplat


def _reference_forward(ten_in, ten_flow):
    ten_in = ten_in.float().cpu()
    ten_flow = ten_flow.float().cpu()
    batch, channels, height, width = ten_in.shape
    output = torch.zeros_like(ten_in)

    for n in range(batch):
        for y in range(height):
            for x in range(width):
                dst_x = x + ten_flow[n, 0, y, x]
                dst_y = y + ten_flow[n, 1, y, x]
                if not torch.isfinite(dst_x) or not torch.isfinite(dst_y):
                    continue
                x0, y0 = int(torch.floor(dst_x)), int(torch.floor(dst_y))
                for target_x, target_y, weight in (
                    (x0, y0, (x0 + 1 - dst_x) * (y0 + 1 - dst_y)),
                    (x0 + 1, y0, (dst_x - x0) * (y0 + 1 - dst_y)),
                    (x0, y0 + 1, (x0 + 1 - dst_x) * (dst_y - y0)),
                    (x0 + 1, y0 + 1, (dst_x - x0) * (dst_y - y0)),
                ):
                    if 0 <= target_x < width and 0 <= target_y < height:
                        output[n, :, target_y, target_x] += ten_in[n, :, y, x] * weight
    return output


def _reference(ten_in, ten_flow, ten_metric, mode, return_norm=False):
    operation, *epsilon = mode.split("-")
    if operation == "avg":
        ten_in = torch.cat((ten_in, torch.ones_like(ten_in[:, :1])), dim=1)
    elif operation == "linear":
        ten_in = torch.cat((ten_in * ten_metric, ten_metric), dim=1)
    elif operation == "softmax":
        metric = ten_metric.exp()
        ten_in = torch.cat((ten_in * metric, metric), dim=1)

    output = _reference_forward(ten_in, ten_flow)
    if operation == "sum":
        return output

    values, norm = output[:, :-1], output[:, -1:]
    epsilon = epsilon[0] if epsilon else "addeps"
    if epsilon == "addeps":
        norm += 1e-7
    elif epsilon == "zeroeps":
        norm = torch.where(norm == 0, 1, norm)
    elif epsilon == "clipeps":
        norm = norm.clamp_min(1e-7)
    return (values, norm) if return_norm else values / norm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="softsplat requires CUDA")
@pytest.mark.parametrize(
    "mode", ["sum", "avg", "linear-zeroeps", "softmax-addeps", "softmax-clipeps"]
)
def test_softsplat_matches_reference(mode):
    torch.manual_seed(1)
    ten_in = torch.randn(2, 3, 5, 7, device="cuda")
    ten_flow = torch.randn(2, 2, 5, 7, device="cuda") * 1.5
    ten_flow[0, 0, 0, 0] = torch.nan
    metric = None if mode in {"sum", "avg"} else torch.randn(2, 1, 5, 7, device="cuda")

    actual = softsplat(ten_in, ten_flow, metric, mode)
    expected = _reference(ten_in, ten_flow, metric, mode)

    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="softsplat requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_softsplat_return_norm_with_low_precision_input(dtype):
    ten_in = torch.randn(1, 4, 4, 6, device="cuda", dtype=dtype)
    ten_flow = torch.randn(1, 2, 4, 6, device="cuda", dtype=dtype)
    metric = torch.randn(1, 1, 4, 6, device="cuda", dtype=dtype)

    actual = softsplat(ten_in, ten_flow, metric, "softmax-zeroeps", return_norm=True)
    expected = _reference(ten_in, ten_flow, metric, "softmax-zeroeps", return_norm=True)

    assert actual[0].dtype == torch.float32
    torch.testing.assert_close(actual[0].cpu(), expected[0], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual[1].cpu(), expected[1], rtol=2e-3, atol=2e-3)
