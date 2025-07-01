# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import threading
from abc import ABC
from functools import lru_cache
from typing import Any, ClassVar
from urllib.parse import urlparse


class LasIO(ABC):
    """Abstract base class for I/O clients.

    This class defines the common interface for all I/O client implementations.
    """

    @classmethod
    def scheme(cls) -> str:
        """Return the scheme identifier for this I/O implementation.

        Returns:
            str: The scheme identifier (e.g., "s3", "gs", "file").

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def mkdirs(self, path: str) -> None:
        """Recursively create a directory and its ancestors.

        Args:
            path: Directory path to create

        Raises:
            FileExistsError: If a file already exists at the specified path
            NotImplementedError: If the subclass does not implement this method
            OtherError: For other I/O related errors
        """
        raise NotImplementedError(f"mkdirs is not supported by {self.__class__.__name__}")

    def rm(self, path: str) -> None:
        """Remove a file or directory recursively.

        If the path does not exist, the operation will be ignored.

        Args:
            path: Path to remove (file or directory)

        Raises:
            NotImplementedError: If the subclass does not implement this method
            OtherError: For other I/O related errors
        """
        raise NotImplementedError(f"rm is not supported by {self.__class__.__name__}")

    def exists(self, path: str) -> bool:
        """Check if a path exists.

        Args:
            path: Path to check

        Returns:
            bool: True if path exists, False otherwise

        Raises:
            NotImplementedError: If the subclass does not implement this method
            OtherError: For other I/O related errors
        """
        raise NotImplementedError(f"exists is not supported by {self.__class__.__name__}")

    def file_size(self, path: str) -> int:
        """Get the size of a file or directory.

        For directories, behavior is implementation-defined (e.g., 4096 for local FS,
        0 for object storage).

        Args:
            path: Path to check size

        Returns:
            int: Size in bytes

        Raises:
            FileNotFoundError: If path does not exist
            NotImplementedError: If the subclass does not implement this method
            OtherError: For other I/O related errors
        """
        raise NotImplementedError(f"file_size is not supported by {self.__class__.__name__}")

    def download_file(self, remote: str, local: str, overwrite: bool = True) -> None:
        """Download a file from remote storage to local filesystem.

        Args:
            remote: Source file path in remote storage
            local: Destination file path in local filesystem (trailing '/' is ignored)
            overwrite: Whether to overwrite existing files (default: True)

        Raises:
            FileNotFoundError: If remote file does not exist
            FileExistsError: If local file exists and overwrite=False
            NotImplementedError: If the subclass does not implement this method
            OtherError: For other I/O related errors
        """
        raise NotImplementedError(f"download_file is not supported by {self.__class__.__name__}")

    def upload_file(self, local: str, remote: str, overwrite: bool = True) -> None:
        """Upload a file from local filesystem to remote storage.

        Args:
            local: Source file path in local filesystem
            remote: Destination file path in remote storage
            overwrite: Whether to overwrite existing files (default: True)

        Raises:
            FileNotFoundError: If local file does not exist
            FileExistsError: If remote file exists and overwrite=False
            NotImplementedError: If the subclass does not implement this method
            OtherError: For other I/O related errors
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


def register_io_client(scheme: str):  # type: ignore
    def decorator(cls: type[LasIO]) -> type[LasIO]:
        LasIOFactory.register(scheme, cls)
        return cls

    return decorator


def mkdirs(uri: str, **kwargs: Any) -> None:
    """Create a new directory and create its ancestors recursively.

    Args:
        uri: The dir path need to be created.

    Raises:
        FileExistsError: if there is a file exists with same path.
        NotImplementedError: if the IO client doesn't support this method.
        OtherError: if any other IO error occurs.
    """
    LasIOFactory.get().get_client(uri, **kwargs).mkdirs(uri)


def rm(uri: str, **kwargs: Any) -> None:
    """Remove a file or dir recursively for the given path.

    Allow the path is not exist.

    Args:
        uri: The dir path or file path need to be removed.

    Raises:
        NotImplementedError: if the IO client doesn't support this method.
        OtherError: if any other IO error occurs.
    """
    LasIOFactory.get().get_client(uri, **kwargs).rm(uri)


def download_file(uri: str, local: str, overwrite: bool = True, **kwargs: Any) -> None:
    """Downloading file from remote datasource to local.

    Args:
        uri: The source file to download. Raise error if source file is not found.
        local: The local file path, trailing `/` in local filename will be ignored.
        overwrite: If the flag is true, will overwrite the existing file or dir,
            but be careful it's not an atomic operation. Raise error if the flag
            is false and the local file or dir exists.

    Raises:
        FileNotFoundError: if the remote file is not exist.
        FileExistsError: if overwrite is False and the local path exists.
        NotImplementedError: if the IO client doesn't support this method.
        OtherError: if any other IO error occurs.
    """
    LasIOFactory.get().get_client(uri, **kwargs).download_file(uri, local, overwrite)


def upload_file(local: str, uri: str, overwrite: bool = True, **kwargs: Any) -> None:
    """Uploading local file to remote.

    Args:
        local: The local file need to be uploaded. Raise error if local file is not found.
        uri: The remote destination file
        overwrite: If the flag is true, will overwrite the existing file or dir,
            but be careful it's not an atomic operation.
            Raise error if the flag is false and the remote file or dir exist.

    Raises:
        FileNotFoundError: if the local file is not exist.
        FileExistsError: if overwrite is False and the remote path exists.
        NotImplementedError: if the IO client doesn't support this method.
        OtherError: if any other IO error occurs.
    """
    LasIOFactory.get().get_client(uri, **kwargs).upload_file(local, uri, overwrite)


def file_size(uri: str, **kwargs: Any) -> int:
    """Return the file size with the given path.

    If the path is a dir, the return value is according to the dir size of target
    source, e.g. 4k for local fs, 0 for object store.

    Args:
        uri: The path need to fetch size.

    Raises:
        FileNotFoundError: if the path is not exist.
        NotImplementedError: if the IO client doesn't support this method.
        OtherError: if any other IO error occurs.
    """
    return LasIOFactory.get().get_client(uri, **kwargs).file_size(uri)


def exists(uri: str, **kwargs: Any) -> bool:
    """Check whether this path exists.

    Args:
        uri: The path need to check.

    Raises:
        NotImplementedError: if the IO client doesn't support this method.
        OtherError: if any other IO error occurs.
    """
    return LasIOFactory.get().get_client(uri, **kwargs).exists(uri)
