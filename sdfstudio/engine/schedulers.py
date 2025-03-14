# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""Scheduler Classes"""

from dataclasses import dataclass, field
from typing import Any, Optional, Type, List

import numpy as np
from torch.optim import Optimizer, lr_scheduler

from sdfstudio.configs.base_config import InstantiateConfig


@dataclass
class SchedulerConfig(InstantiateConfig):
    """Basic scheduler config with self-defined exponential decay schedule"""

    _target: Type = field(default_factory=lambda: ExponentialDecaySchedule)
    lr_final: float = 0.000005
    max_steps: int = 1000000

    # TODO: somehow make this more generic. i dont like the idea of overriding the setup function
    # but also not sure how to go about passing things into predefined torch objects.
    def setup(self, optimizer=None, lr_init=None, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(optimizer, lr_init, self.lr_final, self.max_steps)


class ExponentialDecaySchedule(lr_scheduler.LambdaLR):
    """Exponential learning rate decay function.
    See https://github.com/google-research/google-research/blob/
    fd2cea8cdd86b3ed2c640cbe5561707639e682f3/jaxnerf/nerf/utils.py#L360
    for details.

    Args:
        optimizer: The optimizer to update.
        lr_init: The initial learning rate.
        lr_final: The final learning rate.
        max_steps: The maximum number of steps.
        lr_delay_steps: The number of steps to delay the learning rate.
        lr_delay_mult: The multiplier for the learning rate after the delay.
    """

    config: SchedulerConfig

    def __init__(self, optimizer, lr_init, lr_final, max_steps, lr_delay_steps=0, lr_delay_mult=1.0) -> None:
        def func(step):
            if lr_delay_steps > 0:
                delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                    0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
                )
            else:
                delay_rate = 1.0
            t = np.clip(step / max_steps, 0, 1)
            log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            multiplier = (
                log_lerp / lr_init
            )  # divided by lr_init because the multiplier is with the initial learning rate
            return delay_rate * multiplier

        super().__init__(optimizer, lr_lambda=func)


class DelayerScheduler(lr_scheduler.LambdaLR):
    """Starts with a flat lr schedule until it reaches N epochs then applies a given scheduler"""

    def __init__(
        self,
        optimizer: Optimizer,
        lr_init,  # pylint: disable=unused-argument
        lr_final,  # pylint: disable=unused-argument
        max_steps,  # pylint: disable=unused-argument
        delay_epochs: int = 500,
        after_scheduler: Optional[lr_scheduler.LambdaLR] = None,
    ) -> None:
        def func(step):
            if step > delay_epochs:
                if after_scheduler is not None:
                    multiplier = after_scheduler.lr_lambdas[0](step - delay_epochs)  # type: ignore
                    return multiplier
                return 1.0
            return 0.0

        super().__init__(optimizer, lr_lambda=func)


class DelayedExponentialScheduler(DelayerScheduler):
    """Delayer Scheduler with an Exponential Scheduler initialized afterwards."""

    def __init__(
        self,
        optimizer: Optimizer,
        lr_init,
        lr_final,
        max_steps,
        delay_epochs: int = 200,
    ):
        after_scheduler = ExponentialDecaySchedule(
            optimizer,
            lr_init,
            lr_final,
            max_steps,
        )
        super().__init__(optimizer, lr_init, lr_final, max_steps, delay_epochs, after_scheduler=after_scheduler)


@dataclass
class MultiStepSchedulerConfig(InstantiateConfig):
    """Basic scheduler config with self-defined exponential decay schedule"""

    _target: Type = field(default_factory=lambda: lr_scheduler.MultiStepLR)
    max_steps: int = 1000000

    def setup(self, optimizer=None, lr_init=None, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(
            optimizer,
            milestones=[self.max_steps // 2, self.max_steps * 3 // 4, self.max_steps * 9 // 10],
            gamma=0.33,
        )


@dataclass
class ExponentialSchedulerConfig(InstantiateConfig):
    """Basic scheduler config with self-defined exponential decay schedule"""

    _target: Type = field(default_factory=lambda: lr_scheduler.ExponentialLR)
    decay_rate: float = 0.1
    max_steps: int = 1000000

    def setup(self, optimizer=None, lr_init=None, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(
            optimizer,
            self.decay_rate ** (1.0 / self.max_steps),
        )


@dataclass
class NeuSSchedulerConfig(InstantiateConfig):
    """Basic scheduler config with self-defined exponential decay schedule"""

    _target: Type = field(default_factory=lambda: NeuSScheduler)
    warm_up_end: int = 5000
    learning_rate_alpha: float = 0.05
    max_steps: int = 300000

    def setup(self, optimizer=None, lr_init=None, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(
            optimizer,
            self.warm_up_end,
            self.learning_rate_alpha,
            self.max_steps,
        )


class NeuSScheduler(lr_scheduler.LambdaLR):
    """Starts with a flat lr schedule until it reaches N epochs then applies a given scheduler"""

    def __init__(self, optimizer, warm_up_end, learning_rate_alpha, max_steps) -> None:
        def func(step):
            if step < warm_up_end:
                learning_factor = step / warm_up_end
            else:
                alpha = learning_rate_alpha
                progress = (step - warm_up_end) / (max_steps - warm_up_end)
                learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
            return learning_factor

        super().__init__(optimizer, lr_lambda=func)
        
@dataclass
class MultiStepWarmupSchedulerConfig(InstantiateConfig):
    """Basic scheduler config with self-defined exponential decay schedule"""

    _target: Type = field(default_factory=lambda: MultiStepWarmupScheduler)
    warm_up_end: int = 5000
    milestones: List[int] = field(default_factory=lambda: [300000, 400000, 500000])
    gamma: float = 0.33
    
    def setup(self, optimizer=None, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(
            optimizer,
            self.warm_up_end,
            self.milestones,
            self.gamma
        )

class MultiStepWarmupScheduler(lr_scheduler.LambdaLR):
    """Starts with a flat lr schedule until it reaches N epochs then applies a given scheduler"""

    def __init__(self, optimizer, warm_up_end, milestones, gamma) -> None:
        def func(step):
            if step < warm_up_end:
                learning_factor = step / warm_up_end
            else:
                index = np.searchsorted(milestones, step, side='left')
                learning_factor = gamma ** index
            return learning_factor

        super().__init__(optimizer, lr_lambda=func)
