import json

class Config:
    palette = None
    color_objective = None
    selection = None
    max_population_size = None
    break_condition = None
    cross_over = None
    mutation = None
    select_amount_per_generation = None
    
    @staticmethod
    def load_from_json(json_file):
        if Config.palette is not None:
            return
        with open(json_file) as json_file:
            data = json.load(json_file)
        Config.palette = data['palette']
        Config.color_objective = data['color_objective']
        Config.selection = data['selection']
        Config.max_population_size = data['max_population_size']
        Config.break_condition = data['break_condition']
        Config.cross_over = data['cross_over']
        Config.mutation = data['mutation']
        Config.select_amount_per_generation = data['select_amount_per_generation']
    
    @staticmethod
    def get_palette_color_amount():
        return len(Config.palette)
