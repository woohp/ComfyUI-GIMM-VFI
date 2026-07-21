from comfy_api.latest import ComfyExtension, io

from .nodes import (
    DownloadAndLoadGIMMVFIModel,
    GIMMVFI_interpolate,
    GIMMVFI_interpolate_fps,
)


class GIMMVFIExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            DownloadAndLoadGIMMVFIModel,
            GIMMVFI_interpolate,
            GIMMVFI_interpolate_fps,
        ]


async def comfy_entrypoint() -> GIMMVFIExtension:
    return GIMMVFIExtension()
