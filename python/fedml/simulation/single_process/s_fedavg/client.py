class Client:
    def __init__(
        self,
        client_idx,
        local_training_data,
        local_test_data,
        local_sample_number,
        args,
        device,
        model_trainer,
        global_valid_data,
        class_weight
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.global_valid_data = global_valid_data
        self.local_class_weight = class_weight

    def update_local_dataset(
        self, client_idx, local_training_data, local_test_data, local_sample_number, class_weight
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.local_class_weight = class_weight

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.set_id(self.client_idx)
        self.model_trainer.train(self.local_training_data, self.local_class_weight, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        # metrics = self.model_trainer.test(self.global_valid_data, self.device, self.args)
        return weights

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
