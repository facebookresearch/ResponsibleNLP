# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass
from typing import Callable, Iterable, Optional


@dataclass
class Registration:
    name: str
    factory: Callable
    add_args: Optional[Callable] = None


class Registry:
    @classmethod
    def register(
        cls,
        name: str,
        factory: Callable,
        add_args: Optional[Callable] = None,
    ):
        if not hasattr(cls, "registry"):
            cls.registry = {}
        cls.registry[name] = Registration(
            name=name,
            factory=factory,
            add_args=add_args,
        )

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        assert hasattr(cls, "registry")
        for _, registration in cls.registry.items():
            if registration.add_args is not None:
                registration.add_args(parser)
        return parser

    @classmethod
    def build(cls, name, args):
        assert hasattr(cls, "registry")
        assert name in cls.registry, name
        return cls.registry[name].factory(args)


def batch_iter(it: Iterable, batch_size: int):
    batch = []

    for item in it:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch.clear()

    if batch:
        yield batch
