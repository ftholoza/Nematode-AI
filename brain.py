import numpy as np
import pandas as pd


class Brain:
    def __init__(self, neurons_csv="neurons.csv", synapses_csv="synapses.csv", alpha=0.2):
        self.neurons_csv = neurons_csv
        self.synapses_csv = synapses_csv
        self.alpha = alpha

        self.neurons = pd.read_csv(self.neurons_csv)
        self.synapses = pd.read_csv(self.synapses_csv)

        self.n = len(self.neurons)

        # W[post, pre]
        self.W = np.zeros((self.n, self.n), dtype=float)
        for _, row in self.synapses.iterrows():
            pre = int(row["pre"])
            post = int(row["post"])
            weight = float(row["weight"])
            self.W[post, pre] = weight

        self.x = np.zeros(self.n, dtype=float)

        self.id_to_name = {
            int(row["id"]): row["name"]
            for _, row in self.neurons.iterrows()
        }
        self.name_to_id = {
            row["name"]: int(row["id"])
            for _, row in self.neurons.iterrows()
        }
        self.id_to_type = {
            int(row["id"]): row["type"]
            for _, row in self.neurons.iterrows()
        }

        self.sensory_ids = [
            int(row["id"])
            for _, row in self.neurons.iterrows()
            if row["type"] == "sensory"
        ]
        self.inter_ids = [
            int(row["id"])
            for _, row in self.neurons.iterrows()
            if row["type"] == "inter"
        ]
        self.motor_ids = [
            int(row["id"])
            for _, row in self.neurons.iterrows()
            if row["type"] == "motor"
        ]

    def reset(self):
        self.x = np.zeros(self.n, dtype=float)

    def step(self, sensory_input):
        sensory_input = np.asarray(sensory_input, dtype=float)

        if sensory_input.shape != (self.n,):
            raise ValueError(
                f"sensory_input must have shape ({self.n},), got {sensory_input.shape}"
            )

        total_input = self.W @ self.x + sensory_input
        new_x = (1.0 - self.alpha) * self.x + self.alpha * np.tanh(total_input)

        self.x = new_x
        return new_x.copy(), total_input.copy()

    def get_state(self):
        return self.x.copy()

    def get_neuron_id(self, name):
        return self.name_to_id[name]

    def get_activity_by_name(self, name):
        return float(self.x[self.get_neuron_id(name)])

    def build_sensory_vector(self, values_by_name):
        sensory = np.zeros(self.n, dtype=float)
        for name, value in values_by_name.items():
            neuron_id = self.get_neuron_id(name)
            sensory[neuron_id] = float(value)
        return sensory

    def get_motor_outputs(self):
        return {
            self.id_to_name[i]: float(self.x[i])
            for i in self.motor_ids
        }

    def get_group_state(self, neuron_type):
        valid = {"sensory": self.sensory_ids, "inter": self.inter_ids, "motor": self.motor_ids}
        if neuron_type not in valid:
            raise ValueError("neuron_type must be one of: sensory, inter, motor")

        return {
            self.id_to_name[i]: float(self.x[i])
            for i in valid[neuron_type]
        }

    def set_weight(self, pre_name, post_name, value):
        pre = self.get_neuron_id(pre_name)
        post = self.get_neuron_id(post_name)
        self.W[post, pre] = float(value)

    def get_weight(self, pre_name, post_name):
        pre = self.get_neuron_id(pre_name)
        post = self.get_neuron_id(post_name)
        return float(self.W[post, pre])

    def summary(self):
        lines = []
        lines.append("Brain summary")
        lines.append(f"- neurons: {self.n}")
        lines.append(f"- synapses: {len(self.synapses)}")
        lines.append(f"- alpha: {self.alpha}")
        lines.append("")

        for i in range(self.n):
            lines.append(
                f"{i:02d} | {self.id_to_name[i]:<15} | {self.id_to_type[i]:<7} | x={self.x[i]: .3f}"
            )

        return "\n".join(lines)