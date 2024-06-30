import subprocess
from shutil import rmtree
import os
import time
from typing import List

class Defects4JValidator():
    __defects4j_path = "/home/selab/defects4j/framework/bin/defects4j"

    def command(self, cmd: List[str]):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

        return process

    def notExistPatches(self, patches_dir: str):
        return (os.path.exists(patches_dir) == False)

    def findPatches(self, patches_dir: str):
        return os.listdir(patches_dir)

    def cleanModuleClone(self, module_clone_dir: str):
        if os.path.isdir(module_clone_dir):
            rmtree(module_clone_dir)
        
        os.umask(0)
        os.makedirs(module_clone_dir, mode=0o777)

    def loadModuleClone(self, module_name: str, module_num: str, module_clone_dir: str):
        cmd = [self.__defects4j_path, "checkout", "-p", module_name, "-v", "{}b".format(module_num), "-w", module_clone_dir]
        subprocess.run(cmd)

    def findCompileResults(self, move_dir: str, timeout=100):
        os.chdir(move_dir)

        cmd = [self.__defects4j_path, "compile"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        is_timeout = False
        
        """ start_time = time.time()

        while True:
            if process.poll() is not None:
                break

            seconds = time.time() - start_time
            if timeout and seconds > timeout:
                process.terminate()
                is_timeout = True
            time.sleep(1)

        _, error = process.communicate()"""

        try:
            _, error = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            is_timeout = True

        if is_timeout:
            return ("timeout", [])

        error_results = error.decode('utf-8')
        error_results = error_results.split("\n")
        compile_results = list(filter(None, error_results))

        has_compile_error = False
                
        for compile_result in compile_results:
            if self.__isCompiledError(compile_result):
                has_compile_error = True

        message = "not_compiled" if has_compile_error else "compiled"
        return (message, compile_results)

    # return is_timeout, not_compiled, not_found, passed_testcases
    """ def findTriggerTests(self, module_clone_dir: str, timeout=350):
        os.chdir(module_clone_dir)

        cmd = [self.__defects4j_path, "export", "-p", "tests.trigger"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        is_timeout = False
        
        start_time = time.time()

        while True:
            if process.poll() is not None:
                break

            seconds = time.time() - start_time
            if timeout and seconds > timeout:
                process.terminate()
                is_timeout = True
            time.sleep(1)

        if is_timeout:
            return ("timeout", [])
        
        output, _ = process.communicate()

        output_result = output.decode('utf-8')

        if "FAIL" in output_result:
            return ("not_compiled", [])

        output_results = output_result.split("\n")
        output_results = list(filter(None, output_results))

        output_result_len = len(output_results)

        trigger_tests = []
        for i in range(output_result_len):
            current_trigger_test = output_results[i].strip()
            trigger_tests.append(current_trigger_test)

        return ("not_found" if len(trigger_tests) == 0 else "completed", trigger_tests) """

    # return is_timeout, not_compiled, not_found, passed_testcases
    def findFailedTests(self, module_clone_dir: str, timeout=350):
        os.chdir(module_clone_dir)

        cmd = [self.__defects4j_path, "test", "-r"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        is_timeout = False
        
        """start_time = time.time()

        while True:
            if process.poll() is not None:
                break

            seconds = time.time() - start_time
            if timeout and seconds > timeout:
                process.terminate()
                is_timeout = True
            time.sleep(1) 
            
        output, _ = process.communicate() """
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            is_timeout = True

        if is_timeout:
            return ("timeout", [])

        output_result = output.decode('utf-8')

        # print("{}".format(output))
        # print("{}".format(error))
        if "BUILD FAILED" in output_result:
            return ("not_compiled", [])

        output_results = output_result.split("\n")
        output_results = list(filter(None, output_results))

        output_result_len = len(output_results)

        failed_tests = []

        for i in range(output_result_len):
            if (self.__isFailedTest(output_results[i])):
                for j in range(i+1, output_result_len):
                    current_test = output_results[j].split(" - ")[1].strip()
                    failed_tests.append(current_test)
                break

        return ("not_found" if len(failed_tests) == 0 else "completed", failed_tests)
    
    # return is_timeout, not_compiled, is_plausible, passed_testcases
    def findFailedTestsForPatch(self, module_clone_dir: str, failed_tests: List[str], timeout=350):
        os.chdir(module_clone_dir)

        cmd = [self.__defects4j_path, "test", "-r"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        is_timeout = False
        
        """ start_time = time.time()

        while True:
            if process.poll() is not None:
                break

            seconds = time.time() - start_time
            if timeout and seconds > timeout:
                process.terminate()
                is_timeout = True
            time.sleep(1)

        output, _ = process.communicate() """
    
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            is_timeout = True

        if is_timeout:
            return ("timeout", [])

        output_result = output.decode('utf-8')

        # print("{}".format(output))
        # print("{}".format(error))

        if "BUILD FAILED" in output_result:
            return ("not_compiled", [])
        # print(result)

        if "Failing tests: 0" in output_result:
            return ("passed", [])

        output_results = output_result.split("\n")
        output_results = list(filter(None, output_results))

        # fail less, could be correct
        output_result_len = len(output_results)

        failed_tests = []
        passing_olds = True

        for i in range(output_result_len):
            if (self.__isFailedTest(output_results[i])):
                for j in range(i+1, output_result_len):
                    current_test = output_results[j].split(" - ")[1].strip()
                    if (current_test not in failed_tests):
                        passing_olds = False
                    failed_tests.append(current_test)
                break
        
        message = "passed" if passing_olds or len(failed_tests) == 0 else "not_passed"

        return (message, failed_tests)
    
    def __isCompiledError(self, result: str):
        return not result.endswith("OK")
    
    def __isFailedTest(self, result: str):
        return result.startswith("Failing tests:")
