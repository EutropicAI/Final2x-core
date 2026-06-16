import base64

import pytest

from Final2x_core import SRConfig

from .util import CONFIG_PATH


class Test_SRConfig:
    def test_from_yaml(self) -> None:
        config: SRConfig = SRConfig.from_yaml(CONFIG_PATH)
        print(config)

    def test_from_json_str(self) -> None:
        config: SRConfig = SRConfig.from_yaml(CONFIG_PATH)
        config_json_str = config.model_dump_json()
        config = SRConfig.from_json_str(config_json_str)
        print(config)

    def test_from_base64(self) -> None:
        config: SRConfig = SRConfig.from_yaml(CONFIG_PATH)
        config_json_str = config.model_dump_json()
        b_bytes = base64.b64encode(config_json_str.encode("utf-8"))
        b_str = b_bytes.decode("utf-8")
        config = SRConfig.from_base64(b_str)
        print(config)

    def test_precision_fp16(self) -> None:
        config: SRConfig = SRConfig.from_yaml(CONFIG_PATH)
        config.precision = "fp16"
        config = SRConfig.from_json_str(config.model_dump_json())
        assert config.precision == "fp16"

    def test_precision_bf16(self) -> None:
        config: SRConfig = SRConfig.from_yaml(CONFIG_PATH)
        config.precision = "bf16"
        config = SRConfig.from_json_str(config.model_dump_json())
        assert config.precision == "bf16"

    def test_precision_default(self) -> None:
        config: SRConfig = SRConfig.from_yaml(CONFIG_PATH)
        config_dict = config.model_dump()
        config_dict.pop("precision", None)
        config = SRConfig(**config_dict)
        assert config.precision == "fp32"

    def test_precision_invalid(self) -> None:
        with pytest.raises(ValueError):
            config: SRConfig = SRConfig.from_yaml(CONFIG_PATH)
            config.precision = "invalid"
            SRConfig.from_json_str(config.model_dump_json())

    def test_error_device(self) -> None:
        config: SRConfig
        with pytest.raises(ValueError):
            config = SRConfig.from_yaml(CONFIG_PATH)
            config.device = "wrong_device"
            SRConfig.from_json_str(config.model_dump_json())
