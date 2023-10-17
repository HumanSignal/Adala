import label_studio_sdk

from .base import Dataset, InternalDataFrame
from pydantic import model_validator, SkipValidation
from label_studio_sdk.project import LabelStudioException, Project
from typing import Optional


class LabelStudioDataset(Dataset):

    label_studio_url: str
    label_studio_api_key: str
    label_studio_project_id: int

    ground_truth_column: str = 'ground_truth'

    _project_client: SkipValidation[Project] = None

    @model_validator(mode='after')
    def init_client(self):
        if self._project_client is None:
            client = label_studio_sdk.Client(
                url=self.label_studio_url,
                api_key=self.label_studio_api_key
            )
            self._project_client = client.get_project(id=self.label_studio_project_id)
        return self

    def get_project_info(self):
        return self._project_client.get_params()

    def __len__(self):
        info = self.get_project_info()
        return info['task_number']

    def _tasks_to_df(self, tasks, include_annotations_with_name: str = None):
        indices, records = [], []
        for task in tasks:
            record = task['data']
            if include_annotations_with_name and task['annotations']:
                # TODO: expand more complex annotations
                if len(task['annotations']) > 1:
                    raise NotImplementedError('Multiple annotations are not supported yet')
                annotation = task['annotations'][0]
                annotation_type = annotation['result'][0]['type']
                if annotation_type == 'textarea':
                    annotation_type = 'text'
                if len(annotation['result']) > 1:
                    raise NotImplementedError('Multiple results per annotation are not supported yet')
                label = annotation['result'][0]['value'][annotation_type]
                if isinstance(label, list):
                    if len(label) == 1:
                        label = label[0]
                    else:
                        label = ','.join(sorted(label))
                else:
                    label = str(label)
                record[self.ground_truth_column] = label

            index = task['id']
            records.append(record)
            indices.append(index)
        return InternalDataFrame(records, index=indices)

    def batch_iterator(self, batch_size: int = 100) -> InternalDataFrame:
        page = 1
        while True:
            try:
                data = self._project_client.get_paginated_tasks(page=page, page_size=batch_size)
                yield self._tasks_to_df(data['tasks'])
                page += 1
            # we'll get 404 from API on empty page
            except LabelStudioException as e:
                break

    def get_ground_truth(self, batch: Optional[InternalDataFrame] = None) -> InternalDataFrame:
        if batch is None:
            labeled_tasks = self._project_client.get_labeled_tasks()
            gt = self._tasks_to_df(labeled_tasks, include_annotations_with_name=self.ground_truth_column)
            return gt
        else:
            # TODO: not the most effective method - better to send subset of indices to LS API
            labeled_tasks = self._project_client.get_labeled_tasks()
            gt = self._tasks_to_df(labeled_tasks, include_annotations_with_name=self.ground_truth_column)
            return gt[gt.index.isin(batch.index)]
