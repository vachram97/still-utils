import h5py
import cbf
import os
from abc import ABC, abstractmethod


class ImageBatchWriter:

    def __init__(self, cxi_path=None, h5_path=None):

        # initialize chain of image readers
        self.cxi_writer = CXIWriter(path_to_data=cxi_path)
        self.image_writer = self.cxi_writer

    def write(self, input_list, data):
        for i, image in enumerate(input_list):
            self.image_writer.set_image(image, data[i])
        return


class ImageWriter(ABC):

    @abstractmethod
    def next_writer(self, writer):
        pass

    @abstractmethod
    def set_image(self, path, data):
        pass


class AbstractImageWriter(ImageWriter):

    _next_writer = None

    def next_writer(self, writer):
        self._next_writer = writer
        return writer

    @abstractmethod
    def set_image(self, path, data):
        if self._next_writer:
            return self._next_writer.set_image(path)

        return None


class CXIWriter(AbstractImageWriter):

    def __init__(self, path_to_data=None):
        if path_to_data:
            self.path_to_data = path_to_data
        else:
            self.path_to_data = "/entry_1/data_1/data"

    def set_image(self, path, data):
        if not path.startswith("/"):
            path = os.getcwd() + path
        if (' //' in path) and path.split(' //')[0].endswith(".cxi"):
            return self._set_event(path, data)
        else:
            return super().set_image(path, data)

    def _set_event(self, path, data):
        cxi_path, event = path.split(" //")
        event = int(event)
        with h5py.File(cxi_path, "a") as dataset:
            dset = dataset[self.path_to_data]
            dset[event] = data
            return
