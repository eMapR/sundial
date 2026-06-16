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
        
        self._num_bands = len(self._labels)
        self._num_time_steps = len(self._dates)
        self._chunk_sizes = (len(self._labels), len(self._dates), *self._chunk_sizes[-2:])

    def _consumer(self, consumer_index: int):
        chunk_batch = []
        while (chunk_task := self._chunk_queue.get()) is not None:
            translateY, translateX = chunk_task
            bounds = (translateX, translateY - self._grid_y_size, translateX + self._grid_x_size, translateY)
            
            chunk_stack = []
            self._report_queue.put(("INFO", f"Consumer {consumer_index} rasterizing chunk {translateY, translateX}..."))
            for label in self._labels:
                label_subset = self._geo_proc_data[self._geo_proc_data[self._label_column] == label]
                date_stack = []
                for date in self._dates:
                    date_subset = label_subset[label_subset[self._date_column] == date]
                    if len(date_subset) > 0:
                        arr = rasterizer(date_subset.geometry, bounds, self._chunk_sizes[-2], self._chunk_sizes[-1], 0, 1)
                    else:
                        arr = np.zeros(self._chunk_sizes[-2:])
                    date_stack.append(arr)
                date_stack = np.stack(date_stack)
                chunk_stack.append(date_stack)
            chunk = np.stack(chunk_stack)
            
            self._report_queue.put(("INFO", f"Appending chunk {chunk.shape} to consumer {consumer_index}... {translateY, translateX}"))
            chunk_batch.append((chunk, translateY, translateX))
            self._report_queue.put(("INFO", f"Consumer {consumer_index} contains {len(chunk_batch)} chunks..."))

            if len(chunk_batch) == self._io_limit:
                self._write_array_batch(chunk_batch)
                chunk_batch.clear()
        
        if len(chunk_batch) > 0:
            self._write_array_batch(chunk_batch)
            chunk_batch.clear()
        self._report_queue.put(("INFO", f"Consumer {consumer_index} completed. exiting..."))
        self._result_queue.put(None)
            
            