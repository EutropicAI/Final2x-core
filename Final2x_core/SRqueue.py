from pathlib import Path
from typing import List

import cv2
import numpy as np
from loguru import logger

from Final2x_core.config import SRConfig
from Final2x_core.SRclass import SRWrapper
from Final2x_core.util import PrintProgressLog


def sr_queue(config: SRConfig) -> None:
    """
    Super-resolution queue. Process all RGBA images according to the config.

    :param config: SRConfig
    :return:
    """
    input_path: List[Path] = config.input_path
    output_path: Path = config.output_path / "outputs"
    output_path.mkdir(parents=True, exist_ok=True)  # create output folder
    sr = SRWrapper(config)

    logger.info("Processing------[ 0.0% ]")

    save_format = ".png" if config.save_format is None else config.save_format
    if not save_format.startswith("."):
        logger.warning(f"save_format '{save_format}' does not start with a '.', prepending it.")
        save_format = "." + save_format

    for img_path in input_path:
        base_stem = f"{config.target_scale}x-{Path(img_path).stem}"
        save_path = str(output_path / (base_stem + save_format))
        i: int = 0
        while Path(save_path).is_file():
            logger.warning("Image already exists: " + save_path)
            i += 1
            save_path = str(output_path / (base_stem + "(" + str(i) + ")" + save_format))
            logger.warning("Try to save to: " + save_path)

        if not Path(img_path).is_file():
            logger.error("File not found: " + str(img_path) + ", skip. Save path: " + save_path)
            logger.warning("______Skip_Image______: " + str(img_path))
            PrintProgressLog().skipProgress()

        else:
            alpha_channel = None

            try:
                # The file may not be read correctly.
                # In unix-like system, the Filename Extension is not important.
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

                if len(img.shape) == 2:
                    logger.warning("Grayscale image detected, Convert to RGB image.")
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                elif img.shape[2] == 4:
                    logger.warning("4 channels image detected.")
                    PrintProgressLog().Total += PrintProgressLog().sr_n
                    # Extract alpha channel
                    alpha_channel = img[:, :, 3]
                    # Remove alpha channel from the image
                    img = img[:, :, :3]

                if img is None:
                    raise Exception("Failed to decode image.")
            except Exception as e:
                logger.error(str(e))
                logger.warning("CV2 load image failed: " + str(img_path) + ", skip. ")
                logger.warning("______Skip_Image______: " + str(img_path))
                PrintProgressLog().skipProgress()
                continue

            logger.info("Processing: " + str(img_path) + ", save to: " + save_path)
            img = sr.process(img)

            if alpha_channel is not None:
                # Stack alpha channel into a 3-channel tensor (AAA)
                alpha_tensor = np.dstack((alpha_channel, alpha_channel, alpha_channel))
                # Apply super-resolution to the alpha tensor
                alpha_tensor = sr.process(alpha_tensor)
                # Merge processed RGB channels with processed alpha tensor
                img = np.dstack((img, alpha_tensor[:, :, 0]))

            cv2.imencode(save_format, img)[1].tofile(save_path)

            logger.success("______Process_Completed______: " + str(img_path))
