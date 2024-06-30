import os

class APRDirectory():
    def __init__(self, resources_dir: str):
        self.__resources_dir = resources_dir

    @property
    def resources_dir(self):
        return os.path.join(self.__resources_dir, "APR_Resources")
    
    # Ingredients
    @property
    def ingredients_dir(self):
        return os.path.join(self.resources_dir, "ingredients")

    @property
    def bugs2fix_ingredients_dir(self):
        return os.path.join(self.ingredients_dir, "bugs2fix")
    
    @property
    def defects4j_ingredients_dir(self):
        return os.path.join(self.ingredients_dir, "defects4j")
    @property
    def manysstubs4j_ingredients_dir(self):
        return os.path.join(self.ingredients_dir, "manysstubs4j")

    # Learning
    @property
    def learning_dir(self):
        return os.path.join(self.resources_dir, "learning")

    @property
    def datasets_dir(self):
        return os.path.join(self.learning_dir, "datasets")
    

    # Localization
    @property
    def localization_dir(self):
        return os.path.join(self.resources_dir, "localization")

    @property
    def defects4j_faults_dir(self):
        return os.path.join(self.localization_dir, "defects4j_faults")
    
    @property
    def defects4j_developers_dir(self):
        return os.path.join(self.localization_dir, "defects4j_developers")

    @property
    def manysstubs4j_faults_dir(self):
        return os.path.join(self.localization_dir, "manysstubs4j_faults")
        
    # Repair
    @property
    def repair_dir(self):
        return os.path.join(self.resources_dir, "repair")
    
    @property
    def candidate_patches_dir(self):
        return os.path.join(self.repair_dir, "candidate_patches")
    
    @property
    def combination_patches_dir(self):
        return os.path.join(self.repair_dir, "combination_patches")

    @property
    def combination_patches_mapping_dir(self):
        return os.path.join(self.repair_dir, "combination_patches_mapping")
    
    @property
    def module_clones_dir(self):
        return os.path.join(self.repair_dir, "module_clones")

    @property
    def results_dir(self):
        return os.path.join(self.repair_dir, "results")
    
    # Log
    @property
    def logs_dir(self):
        return os.path.join(self.resources_dir, "logs")
    
    # Binary
    @property
    def bins_dir(self):
        return os.path.join(self.resources_dir, "bins")