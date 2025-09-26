from cccv import ConfigType

from Final2x_core import SRConfig, SRWrapper

from .util import CONFIG_PATH, calculate_image_similarity, compare_image_size, load_image


class Test_TARGETSCALE:
    def test_case_targetscale_positive(self) -> None:
        config: SRConfig = SRConfig.from_yaml(CONFIG_PATH)
        config.pretrained_model_name = ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x
        for t in [7.99999, 1, 2, 2.5, 4, 5.6619, 8, 0.673]:
            config.target_scale = t
            SR = SRWrapper(config=config)
            img1 = load_image()
            img2 = SR.process(img1)
            assert calculate_image_similarity(img1, img2)
            assert compare_image_size(img1, img2, config.target_scale)
