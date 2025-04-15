""" Source: https://github.com/wenhao-gao/mol-opt/blob/main/molopt/base.py 
    Modified to use GuacaMol MPO objectives directly
"""

import torch 
import numpy as np 
import pandas as pd
from typing import List, Union

from guacamol import standard_benchmarks

from hyformer.utils.chemistry import is_valid, canonicalize

# Define the available GuacaMol MPO objectives
GUACAMOL_MPO_TASK_FN = {
    'amlodipine_mpo': standard_benchmarks.amlodipine_rings(),
    'fexofenadine_mpo': standard_benchmarks.hard_fexofenadine(),
    'osimertinib_mpo': standard_benchmarks.hard_osimertinib(),
    'perindopril_mpo': standard_benchmarks.perindopril_rings(),
    'sitagliptin_mpo': standard_benchmarks.sitagliptin_replacement(),
    'ranolazine_mpo': standard_benchmarks.ranolazine_mpo(),
    'zaleplon_mpo': standard_benchmarks.zaleplon_with_other_formula(),
}

# List of all available GuacaMol MPO objectives
GUACAMOL_ORACLES = list(GUACAMOL_MPO_TASK_FN.keys())


class PMOOracle:
    """Predictive Molecular Optimization Oracle for GuacaMol MPO objectives."""

    def __init__(self, name: str, dtype: str = None, max_number_of_calls: int = None, freq_log: int = 100) -> None:
        if name not in GUACAMOL_ORACLES:
            raise ValueError(f"Oracle {name} not available in GuacaMol oracles")
        
        # Get the correct GuacaMol benchmark objective
        self.name = name
        print(f"Initializing GuacaMol benchmark: {name}")
        benchmark = GUACAMOL_MPO_TASK_FN[name]
        print(f"Benchmark type: {type(benchmark)}")
        self.oracle = benchmark.objective
        print(f"Oracle type: {type(self.oracle)}")
        self.dtype = dtype
        self.max_number_of_calls = max_number_of_calls
        self.freq_log = freq_log
        
        self._number_of_calls = 0
        self._buffer = OracleBuffer()
    
    def _to_dtype(self, outputs: List[float]) -> Union[torch.Tensor, np.array, List[float]]:
        if self.dtype == 'pt':
            return torch.tensor(outputs)
        elif self.dtype == 'np':    
            return np.array(outputs)
        else:
            return outputs

    def __len__(self) -> int:
        return self._number_of_calls
    
    def __call__(self, smiles: Union[str, List[str]]) -> float:
        if isinstance(smiles, str):
            smiles = [smiles]

        # Batch canonicalize all SMILES first
        canonical_smiles = [canonicalize(smile, include_stereocenters=False) for smile in smiles]
        
        # Filter out already evaluated molecules
        new_smiles = [smile for smile in canonical_smiles if smile not in self._buffer]
        
        if not new_smiles:
            # All molecules already in buffer, return cached values
            return self._to_dtype([self._buffer[smile][0] for smile in canonical_smiles])
        
        # Batch validate molecules
        valid_mask = [is_valid(smile) for smile in new_smiles]
        
        # Batch score valid molecules
        _oracle_values = []
        for smile, is_valid in zip(new_smiles, valid_mask):
            if self.max_number_of_calls is not None and self._number_of_calls >= self.max_number_of_calls:
                break
                
            if is_valid:
                try:
                    score = self.oracle.score(smile)
                    self._number_of_calls += 1
                except:
                    score = np.nan
            else:
                score = np.nan
                
            _oracle_values.append(score)
            self._buffer.add(smile, score)
        
        # Combine results with cached values
        final_values = []
        for smile in canonical_smiles:
            if smile in self._buffer:
                final_values.append(self._buffer[smile][0])
            else:
                final_values.append(np.nan)
        
        return self._to_dtype(final_values)
    
    def __str__(self) -> str:
        return f"Oracle {self.name}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def save(self, filepath: str):
        idx = []
        molecules = []
        properties = []

        for _, (_molecule, (_prop, _idx)) in enumerate(self._buffer.items()):
            idx.append(_idx)
            molecules.append(_molecule)
            properties.append(_prop)

        _df = pd.DataFrame({
            'idx': idx,
            'molecule': molecules,
            'property': properties
        })
        _df.to_csv(filepath, index=False)
    
    def get_metrics(self, verbose: bool = False):
        _smiles, _scores = self._buffer.get_buffer()
    
        # Calculate basic statistics
        avg_top1 = np.max(_scores) if _scores else 0.0
        avg_top10 = np.mean(sorted(_scores, reverse=True)[:10]) if len(_scores) >= 10 else np.mean(_scores) if _scores else 0.0
        avg_top100 = np.mean(sorted(_scores, reverse=True)[:100]) if len(_scores) >= 100 else np.mean(_scores) if _scores else 0.0
        
        if verbose:
            print(f'{self._number_of_calls}/{self.max_number_of_calls} | '
                f'avg_top1: {avg_top1:.3f} | '
                f'avg_top10: {avg_top10:.3f} | '
                f'avg_top100: {avg_top100:.3f}')

        _finish = self.max_number_of_calls > self._number_of_calls
        
        return {
            "avg_top1": avg_top1, 
            "avg_top10": avg_top10, 
            "avg_top100": avg_top100, 
            "auc_top1": top_auc(self._buffer, 1, _finish, self.freq_log, self.max_number_of_calls),
            "auc_top10": top_auc(self._buffer, 10, _finish, self.freq_log, self.max_number_of_calls),
            "auc_top100": top_auc(self._buffer, 100, _finish, self.freq_log, self.max_number_of_calls),
            "n_oracle": self._number_of_calls,
        }
            
            
class OracleBuffer:
    """Buffer to store molecules and their property values."""

    def __init__(self):
        self._buffer = {}
    
    def __len__(self) -> int:
        return len(self._buffer)
    
    def __contains__(self, item):
        return item in self._buffer
    
    def items(self):
        return self._buffer.items()
    
    def get_buffer(self):
        """Get all molecules and their properties from the buffer."""
        self._sort_buffer()
        smiles = []
        properties = []
        for molecule, (prop, _) in self._buffer.items():
            # Filter out NaN values
            if prop == prop:
                smiles.append(molecule)
                properties.append(prop)
        return smiles, properties
    
    def add(self, smiles: str, property: float):
        """Add a molecule and its property value to the buffer."""
        self._buffer[smiles] = [property, len(self) + 1]
        self._sort_buffer()
    
    def _sort_buffer(self):
        """Sort the buffer by property values in descending order."""
        self._buffer = dict(sorted(self._buffer.items(), key=lambda kv: kv[1][0], reverse=True))

def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls
