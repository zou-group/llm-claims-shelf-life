"""Isolated author code scorer; invoked only in a restricted subprocess."""
import ast
import base64
import importlib.util
import json
import os
import pickle
from pathlib import Path
import sys
import zlib
from shelflife.sourcecode import load_functions

def main():
    repo = Path(sys.argv[1])
    spec = importlib.util.spec_from_file_location('c089_author_eval',repo/'tasks/code/eval_code.py')
    evaluator = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluator)
    cls = load_functions(repo/'tasks/code/task_code.py',['TaskCode'],extra_globals={'Task':object,'check_correctness':evaluator.check_correctness,'ast':ast,'base64':base64,'pickle':pickle,'zlib':zlib})['TaskCode']
    task=object.__new__(cls)
    data=json.load(sys.stdin)
    answer=task.extract_answer(data['text'])
    result=task.evaluator_function(answer,data['item'])
    print('C089_RESULT '+json.dumps({'answer':answer,'evaluation':result}))

if __name__=='__main__': main()
