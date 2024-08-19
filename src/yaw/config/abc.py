from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Type, TypeVar

from yaw.config import default as DEFAULT

Tdict = TypeVar("Tdict", bound="DictRepresentation")
T = TypeVar("T", bound="BaseConfig")


class DictRepresentation(ABC):
    """Base class for an object that can be serialised into a dictionary."""

    @classmethod
    def from_dict(
        cls: Type[Tdict],
        the_dict: dict[str, Any],
        **kwargs: dict[str, Any],  # passing additional constructor data
    ) -> Tdict:
        """Create a class instance from a dictionary representation of the
        minimally required data.

        Args:
            the_dict (:obj:`dict`):
                Dictionary containing the data.
            **kwargs: Additional data needed to construct the class instance.
        """
        return cls(**the_dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the class instance to a dictionary containing a minimal set
        of required data.

        Returns:
            :obj:`dict`
        """
        return asdict(self)


class BaseConfig(DictRepresentation):
    @classmethod
    def create(cls: type[T], **kwargs: Any) -> T:
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
        cls: type[T],
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
