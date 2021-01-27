import os
import re
import pandas as pd
from pathlib import Path
from bella.helper import read_config
from bella.data_types import Target, TargetCollection
from bella.models.target import TargetInd, TargetDepMinus, TargetDep, TargetDepPlus

CONFIG_FP = Path('..', 'config.yaml')
model_zoo_folder = Path(read_config('tdsa_model_zoo', CONFIG_FP))


def load_tdsa(name='td'):
    model = None
    if name == 'tdp2':
        # target Dep Plus 2
        target_dep_2_model_path = Path(model_zoo_folder,
                                       f'{TargetDepPlus.name()}2.h5')
        # Loads the model
        model = TargetDepPlus.load(target_dep_2_model_path)
    if name == 'ti':
        # target Ind
        target_ind_model_path = Path(model_zoo_folder,
                                     f'{TargetInd.name()}.h5')
        # Loads the model
        model = TargetInd.load(target_ind_model_path)
    if name == 'td':
        # target Dep
        target_dep_model_path = Path(model_zoo_folder,
                                     f'{TargetDep.name()}.h5')
        # Loads the model
        model = TargetDep.load(target_dep_model_path)
    if name == 'tdm':
        # target Dep Minus
        target_dep_min_model_path = Path(model_zoo_folder,
                                         f'{TargetDepMinus.name()}.h5')
        # Loads the model
        model = TargetDepMinus.load(target_dep_min_model_path)
    if name == 'tdp1':
        # target Dep Plus 1
        target_dep_1_model_path = Path(model_zoo_folder,
                                       f'{TargetDepPlus.name()}1.h5')
        # Loads the model
        model = TargetDepPlus.load(target_dep_1_model_path)
    return model


def parse_tweets(file_path, **target_collection_kwargs) -> TargetCollection:
    """
        Given file path to the annotated sentiment data it will parse the data and return it as a list of dictionaries.
        :param file_path: File Path to the annotated data
        :param target_collection_kwargs: Keywords to parse to the TargetCollection
                                         constructor that is returned.
        :returns: A TargetCollection containing Target instances.
    """
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError('This file does not exist {}'.format(file_path))
    file_name, _ = os.path.splitext(os.path.basename(file_path))

    sentiment_range = {
        '0': 0,
        '1': 1,
        '2': -1
    }
    sentiment_data = TargetCollection(**target_collection_kwargs)
    sent_dict = {}
    data = pd.read_csv(file_path, index_col=0, error_bad_lines=False)
    for index, row in data.iterrows():
        sent_dict['text'] = str(row['full_text'])
        sent_dict['target'] = str(row['context'])
        sent_dict['sentiment'] = sentiment_range[str(row['sentiment'])]
        text = sent_dict['text'].lower()
        target = sent_dict['target'].lower()
        offsets = [match.span() for match in re.finditer(target, text)]
        print(len(target.split()))
        print(index)
        print(offsets)
        if len(target.split()) > 1:
            joined_target = ''.join(target.split())
            offsets.extend([match.span()
                            for match in re.finditer(joined_target, text)])
        sent_dict['spans'] = [offsets[0]]
        sent_id = file_name + '#{}'.format(index)
        target_id = sent_id + '#{}'.format(index+1)

        sent_dict['sentence_id'] = sent_id
        sent_dict['target_id'] = target_id
        sent_target = Target(**sent_dict)
        sentiment_data.add(sent_target)
        sent_dict = {}
    return sentiment_data
