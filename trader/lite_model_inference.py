from tensorflow import lite
import numpy as np


class LiteModel:
    def __init__(self, path):
        self._interpreter = lite.Interpreter(path)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        self._state_input = list(filter(lambda x: x['name'] == 'state', self._input_details))[0]
        self._portf_input = list(filter(lambda x: x['name'] == 'portf', self._input_details))[0]

    def predict(self, state, portf):
        self._interpreter.set_tensor(self._state_input['index'], state.astype(np.float32))
        self._interpreter.set_tensor(self._portf_input['index'], portf.astype(np.float32))
        self._interpreter.invoke()

        output_data = self._interpreter.get_tensor(self._output_details[0]['index'])
        return output_data

    def input_shapes(self):
        return self._state_input['shape'], self._portf_input['shape']
