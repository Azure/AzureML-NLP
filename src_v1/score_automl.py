# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Score text dataset from model produced by training run."""

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
from azureml.automl.core import inference

from azureml.automl.dnn.nlp.common.constants import ScoringLiterals, OutputLiterals

import numpy as np
import pandas as pd
import json
import pickle
import os

data_sample = PandasParameterType(pd.DataFrame({"reviewText": pd.Series(["example_value"], dtype="object")}))
input_sample = StandardPythonParameterType({'data': data_sample})
method_sample = StandardPythonParameterType("predict")
sample_global_params = StandardPythonParameterType(1.0)

result_sample = NumpyParameterType(np.array(["example_value"]))
output_sample =  StandardPythonParameterType({'Results':result_sample})


def init():
    """This function is called during inferencing environment setup and initializes the model"""
    global model
    model_directory = os.getenv(ScoringLiterals.AZUREML_MODEL_DIR_ENV) + '/model'
    print(' ---- Get Model object Dir {} '.format(os.listdir(model_directory)))

    model_path = os.path.join(model_directory, OutputLiterals.MODEL_FILE_NAME)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)


@input_schema('GlobalParameters', sample_global_params, convert_to_provided_type=False)
@input_schema('Inputs', input_sample)
@output_schema(output_sample)
def run(Inputs, GlobalParameters=1.0) -> str:
    """ This is called every time the endpoint is invoked. It returns the prediction of the input data

    :param Inputs: input data provided by the user
    :type Inputs: dict
    :param GlobalParameters: paramter to select which method to called
    :type: dict
    :return: json string of the result
    :rtype: str
    """
    data = Inputs['data']
    fin_outputs = model.predict(data)
    return {"Results": [str(item) for item in fin_outputs]}