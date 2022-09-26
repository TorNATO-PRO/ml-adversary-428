from data_loader import DataLoader
from config_loader import ConfigLoader
from model import Model

project_config_loader = ConfigLoader()
project_config = project_config_loader.get_config()

project_data_loader = DataLoader(project_config.dataset_config)
project_data = project_data_loader.load_data()

model = Model(project_config)
model.fit_model(project_data.train, project_data.validation)
print(model.make_predictions(project_data.testing))
