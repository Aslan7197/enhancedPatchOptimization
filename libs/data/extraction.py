from dataclasses import dataclass

@dataclass
class ExtractionOption:
    dataset: str = None
    
    bins_dir: str = None
    localization_dir: str = None

    fault_paths_json: str = None

    ingredients_dir: str = None
    extracted_dir: str = None

    logs_dir: str = None