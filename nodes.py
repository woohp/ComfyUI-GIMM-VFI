import os
import torch

from comfy_api.latest import io
import folder_paths
import yaml
import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file

from omegaconf import OmegaConf
from tqdm import tqdm
import cv2

from .gimmvfi.generalizable_INR.gimmvfi_r import GIMMVFI_R
from .gimmvfi.generalizable_INR.gimmvfi_f import GIMMVFI_F

from .gimmvfi.generalizable_INR.configs import GIMMVFIConfig
from .gimmvfi.generalizable_INR.raft import RAFT
from .gimmvfi.generalizable_INR.flowformer.core.FlowFormer.LatentCostFormer.transformer import FlowFormer
from .gimmvfi.generalizable_INR.flowformer.configs.submission import get_cfg
from .gimmvfi.utils.flow_viz import flow_to_image
from .gimmvfi.utils.frame_schedule import fixed_factor_schedule, fps_schedule
from .gimmvfi.utils.utils import InputPadder, RaftArgs, easydict_to_dict

from contextlib import nullcontext

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

script_directory = os.path.dirname(os.path.abspath(__file__))


class DownloadAndLoadGIMMVFIModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DownloadAndLoadGIMMVFIModel",
            display_name="(Down)Load GIMMVFI Model",
            category="GIMM-VFI",
            description="Downloads and loads a GIMM-VFI model.",
            inputs=[
                io.Combo.Input(
                    "model",
                    options=[
                        "gimmvfi_r_arb_lpips_fp32.safetensors",
                        "gimmvfi_f_arb_lpips_fp32.safetensors",
                    ],
                ),
                io.Combo.Input(
                    "precision",
                    options=["fp32", "bf16", "fp16"],
                    default="fp32",
                    optional=True,
                ),
                io.Boolean.Input(
                    "torch_compile",
                    default=False,
                    optional=True,
                    tooltip="Compile part of the model with torch.compile; requires Triton.",
                ),
            ],
            outputs=[
                io.Custom("GIMMVIF_MODEL").Output(display_name="gimmvfi_model")
            ],
        )

    @classmethod
    def execute(cls, model, precision="fp32", torch_compile=False) -> io.NodeOutput:

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[precision]

        download_path = os.path.join(folder_paths.models_dir, 'interpolation', 'gimm-vfi')
        model_path = os.path.join(download_path, model)

        if not os.path.exists(model_path):
            log.info(f"Downloading GMMI-VFI model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="Kijai/GIMM-VFI_safetensors",
                allow_patterns=[f"*{model}*"],
                local_dir=download_path,
                local_dir_use_symlinks=False,
            )

        if "gimmvfi_r" in model:
            config_path = os.path.join(script_directory, "configs", "gimmvfi", "gimmvfi_r_arb.yaml")
            flow_model = "raft-things_fp32.safetensors"
        elif "gimmvfi_f" in model:
            config_path = os.path.join(script_directory, "configs", "gimmvfi", "gimmvfi_f_arb.yaml")
            flow_model = "flowformer_sintel_fp32.safetensors"

        flow_model_path = os.path.join(folder_paths.models_dir, 'interpolation', 'gimm-vfi', flow_model)

        if not os.path.exists(flow_model_path):
            log.info(f"Downloading RAFT model to: {flow_model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="Kijai/GIMM-VFI_safetensors",
                allow_patterns=[f"*{flow_model}*"],
                local_dir=download_path,
                local_dir_use_symlinks=False,
            )
       
            
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config = easydict_to_dict(config)
        config = OmegaConf.create(config)
        arch_defaults = GIMMVFIConfig.create(config.arch)
        config = OmegaConf.merge(arch_defaults, config.arch)

        # load model
        if "gimmvfi_r" in model:
            model = GIMMVFI_R(dtype, config)
             #load RAFT
            raft_args = RaftArgs(
                small=False,
                mixed_precision=False,
                alternate_corr=False
            )
        
            raft_model = RAFT(raft_args)
            raft_sd = load_torch_file(flow_model_path)
            raft_model.load_state_dict(raft_sd, strict=True)
            raft_model.to(dtype).to(device)
            flow_estimator = raft_model
        elif "gimmvfi_f" in model:
            model = GIMMVFI_F(dtype, config)
            cfg = get_cfg()
            flowformer = FlowFormer(cfg.latentcostformer)
            flowformer_sd = load_torch_file(flow_model_path)
            flowformer.load_state_dict(flowformer_sd, strict=True)
            flow_estimator = flowformer.to(dtype).to(device)
            
       
        sd = load_torch_file(model_path)
        model.load_state_dict(sd, strict=False)
      
        model.flow_estimator = flow_estimator
        model = model.eval().to(dtype).to(device)

        if torch_compile:
            model = torch.compile(model)
            
        return io.NodeOutput(model)

# region Interpolate
def _interpolate_schedule(
    gimmvfi_model,
    images,
    ds_factor,
    seed,
    output_flows,
    schedule,
    timestep_batch_size=0,
):
    # GIMM-VFI is not managed by ComfyUI's ModelPatcher, so make room by
    # offloading other managed models before starting its large allocations.
    mm.unload_all_models()
    mm.soft_empty_cache()
    images = images.permute(0, 3, 1, 2)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if images.shape[0] == 1:
        return (images.permute(0, 2, 3, 1).float(), torch.zeros(1, 64, 64, 3))

    device = mm.get_torch_device()
    dtype = gimmvfi_model.dtype
    output_images = [None] * len(schedule)
    output_flows = [None] * len(schedule) if output_flows else None
    entries_by_pair = [[] for _ in range(images.shape[0] - 1)]
    for output_index, (pair, timestep) in enumerate(schedule):
        entries_by_pair[pair].append((output_index, timestep))

    pbar = ProgressBar(images.shape[0] - 1)
    autocast_device = mm.get_autocast_device(device)
    cast_context = (
        torch.autocast(device_type=autocast_device, dtype=dtype)
        if dtype != torch.float32
        else nullcontext()
    )

    with torch.inference_mode(), cast_context:
        for pair, entries in enumerate(tqdm(entries_by_pair)):
            mm.throw_exception_if_processing_interrupted()
            if not entries:
                pbar.update(1)
                continue

            image_0 = images[pair].unsqueeze(0)
            image_1 = images[pair + 1].unsqueeze(0)
            interpolation_entries = []
            for output_index, timestep in entries:
                if timestep <= 1e-7:
                    output_images[output_index] = image_0[0].permute(1, 2, 0).cpu()
                elif timestep >= 1.0 - 1e-7:
                    output_images[output_index] = image_1[0].permute(1, 2, 0).cpu()
                else:
                    interpolation_entries.append((output_index, timestep))

            if interpolation_entries:
                padder = InputPadder(image_0.shape, 32)
                image_0, image_1 = padder.pad(image_0, image_1)
                model_input = torch.cat(
                    (image_0.unsqueeze(2), image_1.unsqueeze(2)), dim=2
                ).to(device, non_blocking=True)
                chunk_size = timestep_batch_size or len(interpolation_entries)
                start = 0

                while start < len(interpolation_entries):
                    mm.throw_exception_if_processing_interrupted()
                    chunk = interpolation_entries[start : start + chunk_size]
                    coordinates = [
                        (
                            gimmvfi_model.sample_coord_input(
                                model_input.shape[0],
                                model_input.shape[-2:],
                                [timestep],
                                device=model_input.device,
                                upsample_ratio=ds_factor,
                            ),
                            None,
                        )
                        for _, timestep in chunk
                    ]
                    timestep_tensors = [
                        timestep
                        * torch.ones(model_input.shape[0], device=model_input.device)
                        for _, timestep in chunk
                    ]
                    try:
                        result = gimmvfi_model(
                            model_input,
                            coordinates,
                            t=timestep_tensors,
                            ds_factor=ds_factor,
                            return_flow_outputs=output_flows is not None,
                        )
                    except torch.OutOfMemoryError:
                        del coordinates, timestep_tensors
                        mm.soft_empty_cache()
                        if len(chunk) == 1:
                            raise
                        chunk_size = max(1, len(chunk) // 2)
                        log.warning(
                            "GIMM-VFI ran out of VRAM; retrying with %d timestep(s) per pass",
                            chunk_size,
                        )
                        continue

                    frames = [padder.unpad(frame) for frame in result["imgt_pred"]]
                    flows = (
                        [padder.unpad(flow) for flow in result["flowt"]]
                        if output_flows is not None
                        else None
                    )

                    for index, (output_index, _) in enumerate(chunk):
                        output_images[output_index] = (
                            frames[index][0].detach().cpu().permute(1, 2, 0)
                        )
                        if output_flows is not None:
                            flow_image = flow_to_image(
                                flows[index]
                                .squeeze()
                                .detach()
                                .cpu()
                                .permute(1, 2, 0)
                                .numpy(),
                                convert_to_bgr=False,
                            )
                            output_flows[output_index] = (
                                torch.from_numpy(flow_image).float() / 255.0
                            )
                    start += len(chunk)
                    del result, frames, flows, coordinates, timestep_tensors

            pbar.update(1)

    image_tensors = torch.stack(output_images).float()
    if output_flows is None:
        flow_tensors = torch.zeros(1, 64, 64, 3)
    else:
        empty_flow = torch.zeros_like(image_tensors[0])
        flow_tensors = torch.stack(
            [flow if flow is not None else empty_flow for flow in output_flows]
        )
    return image_tensors, flow_tensors


class GIMMVFI_interpolate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GIMMVFI_interpolate",
            display_name="GIMM-VFI Interpolate",
            category="PyramidFlowWrapper",
            inputs=[
                io.Custom("GIMMVIF_MODEL").Input("gimmvfi_model"),
                io.Image.Input("images", tooltip="The images to interpolate between"),
                io.Float.Input(
                    "ds_factor", default=1.0, min=0.01, max=1.0, step=0.01
                ),
                io.Int.Input(
                    "interpolation_factor", default=8, min=1, max=100, step=1
                ),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Boolean.Input(
                    "output_flows",
                    default=False,
                    optional=True,
                    tooltip="Output the flow tensors",
                ),
                io.Int.Input(
                    "timestep_batch_size",
                    default=0,
                    min=0,
                    max=100,
                    step=1,
                    optional=True,
                    advanced=True,
                    tooltip="Timesteps per pass; 0 is fastest, lower values use less VRAM",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Image.Output(display_name="flow_tensors"),
            ],
        )

    @classmethod
    def execute(
        cls,
        gimmvfi_model,
        images,
        ds_factor,
        interpolation_factor,
        seed,
        output_flows=False,
        timestep_batch_size=0,
    ) -> io.NodeOutput:
        schedule = fixed_factor_schedule(images.shape[0], interpolation_factor)
        outputs = _interpolate_schedule(
            gimmvfi_model,
            images,
            ds_factor,
            seed,
            output_flows,
            schedule,
            timestep_batch_size,
        )
        return io.NodeOutput(*outputs)


class GIMMVFI_interpolate_fps(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GIMMVFI_interpolate_fps",
            display_name="GIMM-VFI Interpolate (FPS)",
            category="PyramidFlowWrapper",
            description="Resample a clip to a target FPS while preserving its duration.",
            inputs=[
                io.Custom("GIMMVIF_MODEL").Input("gimmvfi_model"),
                io.Image.Input("images", tooltip="The images to interpolate"),
                io.Float.Input(
                    "source_fps", default=24.0, min=0.001, max=1000.0, step=0.001
                ),
                io.Float.Input(
                    "target_fps", default=60.0, min=0.001, max=1000.0, step=0.001
                ),
                io.Float.Input(
                    "ds_factor", default=1.0, min=0.01, max=1.0, step=0.01
                ),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Boolean.Input(
                    "output_flows",
                    default=False,
                    optional=True,
                    tooltip="Output one flow image per frame",
                ),
                io.Int.Input(
                    "timestep_batch_size",
                    default=0,
                    min=0,
                    max=100,
                    step=1,
                    optional=True,
                    advanced=True,
                    tooltip="Timesteps per pass; 0 is fastest, lower values use less VRAM",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Image.Output(display_name="flow_tensors"),
            ],
        )

    @classmethod
    def execute(
        cls,
        gimmvfi_model,
        images,
        source_fps,
        target_fps,
        ds_factor,
        seed,
        output_flows=False,
        timestep_batch_size=0,
    ) -> io.NodeOutput:
        schedule = fps_schedule(images.shape[0], source_fps, target_fps)
        outputs = _interpolate_schedule(
            gimmvfi_model,
            images,
            ds_factor,
            seed,
            output_flows,
            schedule,
            timestep_batch_size,
        )
        return io.NodeOutput(*outputs)
