import numpy as np

from pipeline.utils import rasterizer, ParallelGridAlign


class XarrDateAnnotator(ParallelGridAlign):
    def __init__(self, annotations_path, label_column, date_column, 
                 **kwargs):
        super().__init__(**kwargs)
        self._source_path = annotations_path
        self._label_column = label_column
        self._date_column = date_column
        self._filter_intersect = True
        
        self._labels = sorted(self._geo_proc_data[self._label_column].unique())
        self._dates = sorted(self._geo_proc_data[self._date_column].unique())
        
        self._chunk_sizes = (len(self._labels), len(self._dates), *self._chunk_sizes[-2:])
    
    def _consumer(self, consumer_index: int):
        chunk_batch = []
        while (chunk_task := self._chunk_queue.get()) is not None:
            ty, tx = chunk_task
            bounds = (tx, ty - self._grid_y_size, tx + self._grid_x_size, ty)
            
            chunk = []
            self._report_queue.put(("INFO", f"Consumer {consumer_index} rasterizing chunk {ty, tx}..."))
            for date in self._dates:
                for label in self._labels:
                    mask = self._geo_proc_data[self._label_column] == label
                    mask &= self._geo_proc_data[self._date_column] == date
                    subset = self._geo_proc_data[mask].geometry
                    chunk.append(rasterizer(subset, bounds, self._chunk_sizes[-2], self._chunk_sizes[-1], np.nan, 1))
            chunk = np.stack(chunk)
            chunk = chunk.reshape(len(self._labels), len(self._dates), self._chunk_sizes[-2], self._chunk_sizes[-1])
            
            self._report_queue.put(("INFO", f"Appending chunk {chunk.shape} to consumer {consumer_index} ... {ty, tx}"))
            chunk_batch.append((chunk, ty, tx))
            self._report_queue.put(("INFO", f"Consumer {consumer_index} contains {len(chunk_batch)} chunks..."))

            if len(chunk_batch) == self._io_limit:
                self._write_array_batch(chunk_batch)
                chunk_batch.clear()
        
        if len(chunk_batch) > 0:
            self._write_array_batch(chunk_batch)
            chunk_batch.clear()
        self._report_queue.put(("INFO", f"Consumer {consumer_index} completed. exiting..."))
        self._result_queue.put(None)
            
            