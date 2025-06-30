# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Callable, ClassVar
from urllib.parse import urlparse


class LasIO(ABC):
    """The abstract base class for io client."""

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Initialize io client."""

    @classmethod
    def scheme(cls) -> str:
        """Return the scheme of implementation IO.

        Raises:
        ------
        NotImplementedError
            if the IO client doesn't support this method.
        """
        raise NotImplementedError

    def mkdirs(self, path: str) -> None:
        """Create a new directory and create its ancestors recursively.

        Parameters:
        ----------
        path : str
            The dir path need to be created.

        Raises:
        ------
        FileExistsError
            if there is a file exists with same path.
        NotImplementedError
            if the IO client doesn't support this method.
        OtherError
            if any other IO error occurs.
        """
        raise NotImplementedError(f"mkdirs is not supported by {self.__class__.__name__}")

    def rm(self, path: str) -> None:
        """Remove a file or dir recursively for the given path.

        Allow the path is not exist.

        Parameters:
        ----------
        path : str
            The dir path or file path need to be removed.

        Raises:
        ------
        NotImplementedError
            if the IO client doesn't support this method.
        OtherError
            if any other IO error occurs.
        """
        raise NotImplementedError(f"rm is not supported by {self.__class__.__name__}")

    def exists(self, path: str) -> bool:
        """Check whether this path exists.

        Parameters:
        ----------
        path : str
            The path need to check.

        Raises:
        ------
        NotImplementedError
            if the IO client doesn't support this method.
        OtherError
            if any other IO error occurs.
        """
        raise NotImplementedError(f"exists is not supported by {self.__class__.__name__}")

    def file_size(self, path: str) -> int:
        """Return the file size with the given path.

        If the path is a dir, the return value is according to the dir size of target
        source, e.g. 4k for local fs, 0 for object store.

        Parameters:
        ----------
        path : str
            The path need to fetch size.

        Raises:
        ------
        FileNotFoundError
            if the path is not exist.
        NotImplementedError
            if the IO client doesn't support this method.
        OtherError
            if any other IO error occurs.
        """
        raise NotImplementedError(f"file_size is not supported by {self.__class__.__name__}")

    def download_file(self, remote: str, local: str, overwrite: bool = True) -> None:
        """Downloading file from remote datasource to local.

        Parameters:
        ----------
        remote : str
            The source file to download.
            Raise error if source file is not found.
        local : str
            The local file path, trailing `/` in local filename will be ignored.
        overwrite : bool
            If the flag is true, will overwrite the existing file or dir,
            but be careful it's not an atomic operation.
            Raise error if the flag is false and the local file or dir exists.

        Raises:
        ------
        FileNotFoundError
            if the remote file is not exist.
        FileExistsError
            if overwrite is False and the local path exists.
        NotImplementedError
            if the IO client doesn't support this method.
        OtherError
            if any other IO error occurs.
        """
        raise NotImplementedError(f"download_file is not supported by {self.__class__.__name__}")

    def upload_file(self, local: str, remote: str, overwrite: bool = True) -> None:
        """Uploading local file to remote.

        Parameters:
        ----------
        local : str
            The local file need to be uploaded.
            Raise error if local file is not found.
        remote : str
            The remote destination file
        overwrite : bool
            If the flag is true, will overwrite the existing file or dir,
            but be careful it's not an atomic operation.
            Raise error if the flag is false and the remote file or dir exist.

        Raises:
        ------
        FileNotFoundError
            if the local file is not exist.
        FileExistsError
            if overwrite is False and the remote path exists.
        NotImplementedError
            if the IO client doesn't support this method.
        OtherError
            if any other IO error occurs.
        """
        raise NotImplementedError(f"upload_file is not supported by {self.__class__.__name__}")


class LasIOFactory:
    """The LasIOFactory provides capabilities of create io client via url."""

    _registry: ClassVar[dict[str, type[LasIO]]] = {}
    _clients: ClassVar[dict[str, LasIO]] = {}
    _lock = threading.RLock()

    @classmethod
    @lru_cache(maxsize=1)
    def get(cls) -> LasIOFactory:
        """Get the Las io factory."""
        return cls()

    @staticmethod
    def _parse_scheme(uri: str) -> str:
        parsed = urlparse(uri)
        return parsed.scheme.lower() if parsed.scheme else "file"

    def _get_client_cls(self, scheme: str) -> type[LasIO]:
        if client_cls := self._registry.get(scheme):
            return client_cls

        registered_schemes = ", ".join(self._registry.keys())
        raise ValueError(f"Unsupported scheme '{scheme}'. Available schemes: {registered_schemes}")

    def get_client(self, uri: str, **kwargs: Any) -> LasIO:
        scheme = self._parse_scheme(uri)

        if client := self._clients.get(scheme):
            return client

        client_cls = self._get_client_cls(scheme)

        with self._lock:
            if scheme in self._clients:
                return self._clients.get(scheme)  # type: ignore

            client = client_cls(**kwargs)
            self._clients[scheme] = client
            return client

    @classmethod
    def register(cls, scheme: str, io_cls: type[LasIO]) -> None:
        if not scheme:
            raise ValueError(f"The given {scheme} is not valid.")

        if io_cls is None:
            raise ValueError("The client cannot be None.")

        if not issubclass(io_cls, LasIO):
            raise TypeError(f"{io_cls.__name__} must be a subclass of LasIO.")

        normalized_scheme = scheme.lower()

        with cls._lock:
            if normalized_scheme in cls._registry:
                raise ValueError(f"{scheme} is already registered.")

            cls._registry[normalized_scheme] = io_cls


def register_io_client(scheme: str) -> Callable[[type[LasIO]], type[LasIO]]:
    def decorator(cls: type[LasIO]) -> type[LasIO]:
        LasIOFactory.register(scheme, cls)
        return cls

    return decorator


def mkdirs(uri: str, **kwargs: Any) -> None:
    """Create a new directory and create its ancestors recursively.

    Parameters:
    ----------
    uri : str
        The dir path need to be created.

    Raises:
    ------
    FileExistsError
        if there is a file exists with same path.
    NotImplementedError
        if the IO client doesn't support this method.
    OtherError
        if any other IO error occurs.
    """
    LasIOFactory.get().get_client(uri, **kwargs).mkdirs(uri)


def rm(uri: str, **kwargs: Any) -> None:
    """Remove a file or dir recursively for the given path.

    Allow the path is not exist.

    Parameters:
    ----------
    uri : str
        The dir path or file path need to be removed.

    Raises:
    ------
    NotImplementedError
        if the IO client doesn't support this method.
    OtherError
        if any other IO error occurs.
    """
    LasIOFactory.get().get_client(uri, **kwargs).rm(uri)


def download_file(uri: str, local: str, overwrite: bool = True, **kwargs: Any) -> None:
    """Downloading file from remote datasource to local.

    Parameters:
    ----------
    uri : str
        The source file to download.
        Raise error if source file is not found.
    local : str
        The local file path, trailing `/` in local filename will be ignored.
    overwrite : bool
        If the flag is true, will overwrite the existing file or dir,
        but be careful it's not an atomic operation.
        Raise error if the flag is false and the local file or dir exists.

    Raises:
    ------
    FileNotFoundError
        if the remote file is not exist.
    FileExistsError
        if overwrite is False and the local path exists.
    NotImplementedError
        if the IO client doesn't support this method.
    OtherError
        if any other IO error occurs.
    """
    LasIOFactory.get().get_client(uri, **kwargs).download_file(uri, local, overwrite)


def upload_file(local: str, uri: str, overwrite: bool = True, **kwargs: Any) -> None:
    """Uploading local file to remote.

    Parameters:
    ----------
    local : str
        The local file need to be uploaded.
        Raise error if local file is not found.
    uri : str
        The remote destination file
    overwrite : bool
        If the flag is true, will overwrite the existing file or dir,
        but be careful it's not an atomic operation.
        Raise error if the flag is false and the remote file or dir exist.

    Raises:
    ------
    FileNotFoundError
        if the local file is not exist.
    FileExistsError
        if overwrite is False and the remote path exists.
    NotImplementedError
        if the IO client doesn't support this method.
    OtherError
        if any other IO error occurs.
    """
    LasIOFactory.get().get_client(uri, **kwargs).upload_file(local, uri, overwrite)


def file_size(uri: str, **kwargs: Any) -> int:
    """Return the file size with the given path.

    If the path is a dir, the return value is according to the dir size of target
    source, e.g. 4k for local fs, 0 for object store.

    Parameters:
    ----------
    uri : str
        The path need to fetch size.

    Raises:
    ------
    FileNotFoundError
        if the path is not exist.
    NotImplementedError
        if the IO client doesn't support this method.
    OtherError
        if any other IO error occurs.
    """
    return LasIOFactory.get().get_client(uri, **kwargs).file_size(uri)


def exists(uri: str, **kwargs: Any) -> bool:
    """Check whether this path exists.

    Parameters:
    ----------
    uri : str
        The path need to check.

    Raises:
    ------
    NotImplementedError
        if the IO client doesn't support this method.
    OtherError
        if any other IO error occurs.
    """
    return LasIOFactory.get().get_client(uri, **kwargs).exists(uri)
