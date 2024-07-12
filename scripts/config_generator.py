import itertools

class ConfigGenerator:
    def __init__(self, sweep_config):
        self.sweep_config = sweep_config

    def generate_combinations(self, config):
        # Extract fixed parameters
        fixed_params = {key: config[key] for key in config if key != 'kwargs'}

        # Extract dynamic parameters
        dynamic_params = config.get('kwargs', {})

        # Generate all combinations of dynamic parameters
        keys, values = zip(*dynamic_params.items()) if dynamic_params else ([], [])
        value_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)] if values else [{}]

        return fixed_params, value_combinations

    def get_all_configs(self):
        # Generate combinations for model and dataset
        model_fixed, model_combinations = self.generate_combinations(self.sweep_config['model'])
        dataset_fixed, dataset_combinations = self.generate_combinations(self.sweep_config['dataset'])

        # Combine model and dataset configurations
        all_configs = []
        for model_name in model_fixed['name']:
            for dataset_name in dataset_fixed['name']:
                for model_kwargs in model_combinations:
                    for dataset_kwargs in dataset_combinations:
                        config = {
                            "model": {
                                "name": model_name,
                                "kwargs": model_kwargs
                            },
                            "dataset": {
                                "name": dataset_name,
                                "kwargs": dataset_kwargs
                            }
                        }
                        all_configs.append(config)

        return all_configs