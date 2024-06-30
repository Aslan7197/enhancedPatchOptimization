import os
from dataclasses import dataclass

@dataclass
class FaultPath:
    module_id: str = None
    module_dir: str = None

    file_name: str = None
    file_ext: str = ".java"
    
    buggy_dir: str = "F_dir"
    fixed_dir: str = "P_dir"

    file_path: str = None

class DatasetFaultPath():
    def __init__(self, module_id, file_name, file_ext, buggy_dir, fixed_dir, module_dir=None, file_path=None):
        self.module_id = module_id
        self.file_name = file_name
        self.file_ext = file_ext
        self.buggy_dir = buggy_dir
        self.fixed_dir = fixed_dir

        self.module_dir = module_dir
        self.file_path = file_path

    def getBuggyRootDir(self):
        buggy_root_dir = os.path.join(self.module_id, self.buggy_dir)

        if self.module_dir is not None and len(self.module_dir) > 0:
            buggy_root_dir = os.path.join(buggy_root_dir, self.module_dir)

        return buggy_root_dir
	
    def getBuggySubDir(self):
        return os.path.join(self.getBuggyRootDir(), self.file_name)
    
    def getBuggyFullPath(self):
        return self.getBuggySubDir() + self.file_ext
    
    def getFixedRootDir(self):
        fixed_root_dir = os.path.join(self.module_id, self.fixed_dir)

        if self.module_dir is not None and len(self.module_dir) > 0:
            fixed_root_dir = os.path.join(fixed_root_dir, self.module_dir)

        return fixed_root_dir
	
    def getFixedSubDir(self):
        return os.path.join(self.getFixedRootDir(), self.file_name)
    
    def getFixedFullPath(self):
        return self.getFixedSubDir() + self.file_ext