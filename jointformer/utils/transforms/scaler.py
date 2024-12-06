


class Scaler:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, x):
        return (x - self.mean) / self.std
    
    def inverse_transform(self, x):
        return x * self.std + self.mean

    @classmethod
    def from_config(cls, config: dict) -> "Scaler":
        return cls(mean=config.get('mean'), std=config.get('std'))
    