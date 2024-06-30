from dataclasses import dataclass
import os
from typing import Any, Dict, List

# 새로 작성
@dataclass()
class Defects4JBuggyLine():
    bogy: str

    buggy_start_line: int
    buggy_end_line: int

    is_omission: bool    
    is_duplicated: bool
    is_empty: bool
    is_recuplicated: bool

    replaced_bogy: str = None
    replaced_start_line: int = None
    replaced_end_line: int = None

@dataclass
class Defects4JMultiChunk():
    module_id: str
    module_name: str
    module_num: str
    module_dir: str

    buggy_dir: str

    file_key: str
    file_ext: str
    file_name: str

    start_line: int = None
    end_line: int = None

    buggy_lines: Dict[str, List[Defects4JBuggyLine]] = None
    buggy_line_total: int = None
    chunk_total: int = None

    def getBuggyRootDir(self):
        return os.path.join(self.module_id, self.buggy_dir, self.module_dir)
	
    def getBuggySubDir(self):
        return os.path.join(self.getBuggyRootDir(), self.file_name)
    
    def getBuggyFullPath(self):
        return self.getBuggySubDir() + self.file_ext
    
    def getFixedRootDir(self):
        return os.path.join(self.module_id, self.module_dir)
	
    def getFixedSubDir(self):
        return os.path.join(self.getFixedRootDir(), self.file_name)
    
    def getFixedFullPath(self):
        return self.getFixedSubDir() + self.file_ext
    
    def getFileFullName(self):
        return self.file_name + self.file_ext

@dataclass()
class Defects4JModule():
    module_id: str
    module_name: str
    module_num: str

    chunk_total: int

    buggy_line_total: int
    buggy_lines: Dict[str, List[Defects4JBuggyLine]]

    multi_chunk_total: int
    multi_chunks: Dict[str, List[Defects4JMultiChunk]]

    candidate_patches: List[Any] = None