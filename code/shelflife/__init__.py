"""Standardized reproduction pipeline for the shelf life of LLM findings.

Layers, from the bottom:
  models / policies / transport / ledger   request adaptation, HTTP, budget ledger, raw-record persistence
  prompts / program / registry             verbatim prompt templates, prompting strategies, per-claim configs
  runner / replay / stats                  parallel execution with resume, re-scoring saved outputs, statistics
  dspy_bridge                              optional DSPy front end over the same ledgered transport
"""
import os
import sys

# experiments/ is read-only: importing the authors' code or running their venvs must not leave bytecode caches there.
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from .prompts import PromptTemplate, Source
from .program import LM, Program, SingleTurn
from .registry import Experiment, PaperTarget, PriorRun, Variant, get, list_experiments, register

__all__ = ["Experiment", "LM", "PaperTarget", "PriorRun", "Program", "PromptTemplate", "SingleTurn", "Source",
           "Variant", "get", "list_experiments", "register"]
