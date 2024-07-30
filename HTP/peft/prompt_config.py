# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

@dataclass
class PromptTuningConfig:

    num_prefix_tokens: Optional[int] = field(default=None, metadata={"help": "Number of prefix tokens"})
    hidden_size: Optional[int] = field(
        default=None, metadata={"help": "The hidden embedding dimension of the transformer model"}
    )
    tensor_parallel_degree: int = field(default=-1, metadata={"help": ("1 for not use tensor parallel")})
    dtype: Optional[str] = field(default=None, metadata={"help": "The data type of tensor"})

   
    def __dict__(self):
        return asdict(self)

    def to_dict(self):
        return self.__dict__

    def save_pretrained(self, save_directory):
        r"""
        This method saves the configuration of your adapter model in a directory.
        Args:
            save_directory (`str`):
                The directory where the configuration will be saved.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        output_dict = self.__dict__
        output_path = os.path.join(save_directory, 'pt_config.json')

        # save it
        with open(output_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))