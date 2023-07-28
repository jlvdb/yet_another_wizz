from __future__ import annotations

from abc import abstractmethod
from typing import Any, Type, TypeVar

from yaw.config import DEFAULT
from yaw.core.abc import DictRepresentation

T = TypeVar("T", bound="BaseConfig")


class BaseConfig(DictRepresentation):
    @classmethod
    def create(cls: Type[T], **kwargs: Any) -> T:
        """Create a new configuration object.

        By default this is an alias for :meth:`__init__`. Configuration classes
        that are hierarchical (i.e. contain configuration objects as attributes)
        implement this method to provide a single constructer for its own and
        its subclasses parameters.
        """
        return cls(**kwargs)

    @abstractmethod
    def modify(self: T, **kwargs: Any | DEFAULT.NotSet) -> T:
        """Create a copy of the current configuration with updated parameter
        values.

        The method arguments are identical to :meth:`create`. Values that should
        not be modified are by default represented by the special value
        :obj:`~yaw.config.default.NotSet`.
        """
        conf_dict = self.to_dict()
        conf_dict.update(
            {key: value for key, value in kwargs.items() if value is not DEFAULT.NotSet}
        )
        return self.__class__.from_dict(conf_dict)

    @classmethod
    def from_dict(
        cls: Type[T],
        the_dict: dict[str, Any],
        **kwargs: dict[str, Any],  # passing additional constructor data
    ) -> T:
        """Create a class instance from a dictionary representation of the
        minimally required data.

        Args:
            the_dict (:obj:`dict`):
                Dictionary containing the data.
            **kwargs: Additional data needed to construct the class instance.
        """
        return cls.create(**the_dict)
